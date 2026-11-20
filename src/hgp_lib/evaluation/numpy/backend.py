import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy import ndarray
from numpy.typing import DTypeLike

from ...rules import Rule
from ...utils.validation import validate_batching
from ..backend import EvaluationBackend, Evaluator
from ..dataset import Dataset
from ..scorers import ScoreFn
from .kernels import bind_kernels
from .predict import EVALUATIONS

if os.getenv("HGP_LOW_MEMORY") is not None:
    warnings.warn(
        "HGP_LOW_MEMORY is deprecated and ignored; it will be removed in a future 2.x "
        "release. Choose the evaluation with BooleanGPConfig(backend=NumpyBackend("
        "evaluation=...)) instead: HGP_LOW_MEMORY=1 corresponds to "
        'evaluation="inplace".',
        FutureWarning,
        stacklevel=2,
    )


def check_evaluation(evaluation: str) -> None:
    """
    Validate the ``evaluation`` option of a backend.

    Args:
        evaluation (str): ``"block"``, ``"inplace"`` or ``"fused"``.

    Raises:
        ValueError: If ``evaluation`` is not one of the three values.
    """
    if evaluation not in EVALUATIONS:
        valid = ", ".join(repr(name) for name in EVALUATIONS)
        raise ValueError(f"evaluation must be one of {valid}, is {evaluation!r}")


# TODO: Bit-packing. Store each feature column as packed uint64 words (64 rows per
#  word), evaluate rules with bitwise operations on the words and count with
#  np.bitwise_count (NumPy >= 2.0). This could be an option here or a separate backend.
#  Weighted counts (merge_rows=True) need the result unpacked first. Benchmark it on
#  small and large datasets before choosing a default.
@dataclass(frozen=True, kw_only=True)
class NumpyBackend(EvaluationBackend):
    """
    Evaluate rules with NumPy. The default backend.

    The default values were chosen based on benchmarking.

    Attributes:
        order (str): Memory layout of bound data, ``"F"`` (column-major) or ``"C"``
            (row-major). Rules read feature columns, which are contiguous in ``"F"``
            order. Data is converted once when binding, after merging. `predict` never
            copies the data. Default: `"F"`.
        evaluation (str): How rules are evaluated, see `hgp_lib.evaluation.numpy.predict`:
            ``"block"`` gathers the literal children of an operator into one block and
            reduces it; ``"inplace"`` updates one result buffer per operator in place;
            ``"fused"`` does the same, with negated literals fused into the update.
            Default: `"fused"`.
        batched (bool): Stack the predictions of ``batch_size`` rules and score them
            with one call, instead of one rule at a time. Applies to the built-in
            scorers and to scoring functions with a ``batched`` method; other functions
            take one prediction per call. Default: `False`.
        batch_size (int | None): Number of rules per batch when ``batched`` is
            ``True``. ``None`` scores all rules in one batch. A batch needs a boolean
            block of ``batch_size`` times the number of bound rows, plus a copy of it in
            the count dtype for weighted data. Setting it requires ``batched=True``.
            Default: `None`.
        dtype (DTypeLike): Float dtype of the scores. Counts stay exact: weighted
            counts switch to ``float64`` when ``dtype`` cannot represent them exactly.
            Stored as a ``numpy.dtype``. Default: `numpy.float32`.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset, NumpyBackend
        >>> from hgp_lib.rules import Literal, Or
        >>> data = np.array([[True, False], [False, True], [False, False]])
        >>> evaluator = NumpyBackend().bind(Dataset(data, np.array([True, True, False])))
        >>> evaluator.score([Literal(value=0), Or([Literal(value=0), Literal(value=1)])]).tolist()
        [0.6666666865348816, 1.0]
        >>> evaluator.confusion_matrix(Literal(value=1))
        (1, 0, 1, 1)
    """

    order: Literal["F", "C"] = "F"
    evaluation: Literal["block", "inplace", "fused"] = "fused"
    batched: bool = False
    batch_size: int | None = None
    dtype: DTypeLike = np.float32

    def __post_init__(self):
        if self.order not in ("F", "C"):
            raise ValueError(f"order must be 'F' or 'C', is {self.order!r}")
        check_evaluation(self.evaluation)
        validate_batching(self.batched, self.batch_size)
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise TypeError(
                f"dtype must be a float dtype, is {self.dtype!r}"
            ) from error
        if not np.issubdtype(dtype, np.floating):
            raise ValueError(f"dtype must be a float dtype, is {dtype}")
        object.__setattr__(self, "dtype", dtype)

    def _bind(
        self, dataset: Dataset, score_fn: ScoreFn, merge_rows: bool
    ) -> "NumpyEvaluator":
        data = np.asarray(dataset.data, order=self.order)
        return NumpyEvaluator(self, dataset._replace(data=data), score_fn, merge_rows)

    def predict(self, rule: Rule, data: ndarray) -> ndarray:
        return EVALUATIONS[self.evaluation](rule, data)

    def from_numpy(self, data: ndarray) -> ndarray:
        return data

    def to_numpy(self, values: Any) -> ndarray:
        return np.asarray(values)


class NumpyEvaluator(Evaluator):
    """
    `Evaluator` of the `NumpyBackend`. Created by `NumpyBackend.bind`.

    ``dataset.data`` is stored in the backend's memory order.
    """

    def __init__(
        self,
        backend: NumpyBackend,
        dataset: Dataset,
        score_fn: ScoreFn,
        merge_rows: bool,
    ):
        super().__init__(backend, dataset, score_fn, merge_rows)
        self._evaluate = EVALUATIONS[backend.evaluation]
        self._dtype = backend.dtype
        self._kernel, self._confusion = bind_kernels(dataset, score_fn, backend.dtype)
        # Kernels without a batched method take one prediction per call.
        self._batched = backend.batched and hasattr(self._kernel, "batched")
        self._batch_size = backend.batch_size

    def score(self, rules: Sequence[Rule]) -> ndarray:
        if self._batched:
            return self._score_batched(rules)
        data = self.dataset.data
        evaluate_rule = self._evaluate
        kernel = self._kernel
        return np.fromiter(
            (kernel(evaluate_rule(rule, data)) for rule in rules),
            dtype=self._dtype,
            count=len(rules),
        )

    def _score_batched(self, rules: Sequence[Rule]) -> ndarray:
        data = self.dataset.data
        evaluate_rule = self._evaluate
        score_batch = self._kernel.batched
        size = self._batch_size or max(1, len(rules))
        # One row per rule, reused by every batch.
        block = np.empty((min(size, len(rules)), len(data)), dtype=bool)
        scores = np.empty(len(rules), dtype=self._dtype)
        for start in range(0, len(rules), size):
            batch = rules[start : start + size]
            predictions = block[: len(batch)]
            for row, rule in zip(predictions, batch):
                row[...] = evaluate_rule(rule, data)
            scores[start : start + len(batch)] = score_batch(predictions)
        return scores

    def confusion_matrix(self, rule: Rule) -> tuple[int, int, int, int]:
        return self._confusion(self._evaluate(rule, self.dataset.data))
