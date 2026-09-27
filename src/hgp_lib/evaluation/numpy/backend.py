import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy import ndarray

from ...rules import Rule
from ...utils.validation import validate_batching
from ...utils.warnings import warn_once
from ..backend import EvaluationBackend, Evaluator
from ..dataset import Dataset
from ..scorers import Scorer
from .kernels import ConfusionCounts, bind_kernel
from .predict import evaluate, evaluate_low_memory


def _low_memory_from_environment() -> bool:
    value = os.getenv("HGP_LOW_MEMORY")
    if value is None:
        return True
    # FutureWarning, because a DeprecationWarning raised inside library code is hidden.
    warn_once(
        FutureWarning(
            "HGP_LOW_MEMORY is deprecated and will be removed in a future 2.x release. "
            "Use BooleanGPConfig(backend=NumpyBackend(low_memory=...)) instead. "
            "low_memory=True is the default."
        ),
        stacklevel=4,
    )
    return value == "1"


@dataclass(frozen=True, kw_only=True)
class NumpyBackend(EvaluationBackend):
    """
    Evaluate rules with NumPy. The default backend.

    Options only change speed and memory use, never results. In the repository's
    backend benchmarks, ``order="F"`` with ``low_memory=True`` was the fastest in every
    scenario. ``batched`` is off by default because it only helped on small data.

    Attributes:
        order (str): Memory layout of bound data, ``"F"`` (column-major) or ``"C"``
            (row-major). Rules read feature columns, which are contiguous in ``"F"``
            order (1.1x to 14x faster in the benchmarks). Data is converted once when
            binding, after merging. `predict` never copies the data. Default: `"F"`.
        low_memory (bool | None): ``True`` evaluates each operator in one buffer updated
            in place, folding literal children in without temporary arrays. ``False``
            gathers the literal children into one block and reduces it, which needs fewer
            NumPy calls but a temporary block per operator. With ``"F"`` order, ``True``
            was 1.3x to 2.2x faster. ``None`` means ``True``, unless the deprecated
            ``HGP_LOW_MEMORY`` environment variable is set (``"1"`` means ``True``).
            Default: `None`.
        batched (bool): Stack the predictions of ``batch_size`` rules and score them
            with one call, instead of one rule at a time. Only applies to the built-in
            scorers; custom scorers take one prediction per call. With the other
            defaults and one batch, it was 4% to 23% faster on the merged training rows
            of the benchmark datasets (a few hundred to a few thousand rows), and up to
            44% slower on the same rows repeated 10 times. Default: `False`.
        batch_size (int | None): Number of rules per batch when ``batched`` is
            ``True``. ``None`` scores all rules in one batch. A batch needs a boolean
            block of ``batch_size`` times the number of bound rows, plus a ``float64``
            copy of it for weighted data. Setting it requires ``batched=True``.
            Default: `None`.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset, NumpyBackend
        >>> from hgp_lib.evaluation.scorers import resolve_scorer
        >>> from hgp_lib.rules import Literal, Or
        >>> data = np.array([[True, False], [False, True], [False, False]])
        >>> evaluator = NumpyBackend().bind(Dataset(data, np.array([1, 1, 0])), resolve_scorer())
        >>> evaluator.score([Literal(value=0), Or([Literal(value=0), Literal(value=1)])]).tolist()
        [0.6666666666666666, 1.0]
        >>> evaluator.confusion_matrix(Literal(value=1))
        (1, 0, 1, 1)
    """

    order: Literal["F", "C"] = "F"
    low_memory: bool | None = None
    batched: bool = False
    batch_size: int | None = None

    def __post_init__(self):
        if self.order not in ("F", "C"):
            raise ValueError(f"order must be 'F' or 'C', is {self.order!r}")
        if self.low_memory is None:
            object.__setattr__(self, "low_memory", _low_memory_from_environment())
        if not isinstance(self.low_memory, bool):
            raise TypeError(f"low_memory must be a bool, is {type(self.low_memory)}")
        validate_batching(self.batched, self.batch_size)

    def _bind(self, dataset: Dataset, scorer: Scorer) -> "NumpyEvaluator":
        data = np.asarray(dataset.data, dtype=bool, order=self.order)
        return NumpyEvaluator(
            dataset._replace(data=data),
            scorer,
            self.low_memory,
            self.batched,
            self.batch_size,
        )

    def predict(self, rule: Rule, data: ndarray) -> ndarray:
        if self.low_memory:
            return evaluate_low_memory(rule, data)
        return evaluate(rule, data)


class NumpyEvaluator(Evaluator):
    """
    `Evaluator` of the `NumpyBackend`. Created by `NumpyBackend.bind`.

    ``dataset.data`` is stored in the backend's memory order.
    """

    def __init__(
        self,
        dataset: Dataset,
        scorer: Scorer,
        low_memory: bool,
        batched: bool,
        batch_size: int | None,
    ):
        super().__init__(dataset, scorer)
        self._evaluate = evaluate_low_memory if low_memory else evaluate
        self._kernel = bind_kernel(dataset, scorer)
        self._confusion = ConfusionCounts(dataset)
        # Only the built-in kernels score a batch; custom scorers take one prediction.
        self._batched = batched and hasattr(self._kernel, "batched")
        self._batch_size = batch_size

    def score(self, rules: Sequence[Rule]) -> ndarray:
        if self._batched:
            return self._score_batched(rules)
        data = self.dataset.data
        evaluate_rule = self._evaluate
        kernel = self._kernel
        return np.fromiter(
            (kernel(evaluate_rule(rule, data)) for rule in rules),
            dtype=np.float64,
            count=len(rules),
        )

    def _score_batched(self, rules: Sequence[Rule]) -> ndarray:
        data = self.dataset.data
        evaluate_rule = self._evaluate
        score_batch = self._kernel.batched
        size = self._batch_size or max(1, len(rules))
        # One row per rule, reused by every batch.
        block = np.empty((min(size, len(rules)), len(data)), dtype=bool)
        scores = np.empty(len(rules))
        for start in range(0, len(rules), size):
            batch = rules[start : start + size]
            predictions = block[: len(batch)]
            for row, rule in zip(predictions, batch):
                row[...] = evaluate_rule(rule, data)
            scores[start : start + len(batch)] = score_batch(predictions)
        return scores

    def confusion_matrix(self, rule: Rule) -> tuple[int, int, int, int]:
        return self._confusion(self._evaluate(rule, self.dataset.data))
