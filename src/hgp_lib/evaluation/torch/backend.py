import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from ...rules import Rule
from ...utils.validation import validate_batching
from ..backend import EvaluationBackend, Evaluator
from ..dataset import Dataset
from ..numpy.backend import check_evaluation
from ..scorers import ScoreFn
from .dataset import TorchDataset, column_major
from .kernels import bind_kernels
from .predict import EVALUATIONS


@dataclass(frozen=True, kw_only=True)
class TorchBackend(EvaluationBackend):
    """
    Evaluate rules with PyTorch, on the CPU or on a GPU.

    The rows, predictions and scores stay on ``device``: the evaluator's dataset is a
    `TorchDataset`, and custom scoring functions receive tensors (labels, predictions
    and ``sample_weight``), so they must work with tensors. Values reach NumPy through `to_numpy`, once per population.

    The default values were chosen based on benchmarking.

    Attributes:
        device (torch.device | str | None): Where the bound rows are stored and rules are
            evaluated, for example ``"cuda"``, ``"cuda:1"`` or ``"mps"``. ``None`` means
            ``"cpu"``. Stored as a ``torch.device``. Default: `None`.
        evaluation (str): How rules are evaluated, as for `NumpyBackend`: ``"block"``,
            ``"inplace"`` or ``"fused"``. Default: `"fused"`.
        batched (bool): Stack the predictions of ``batch_size`` rules and score them
            with one call, instead of one rule at a time. Applies to the built-in
            scorers and to scoring functions with a ``batched`` method; other functions
            take one prediction per call. Default: `False`.
        batch_size (int | None): Number of rules per batch when ``batched`` is
            ``True``. ``None`` scores all rules in one batch. A batch needs a boolean
            block of ``batch_size`` times the number of bound rows on ``device``, plus a
            copy of it in the count dtype for weighted data. Setting it requires
            ``batched=True``. Default: `None`.
        dtype (torch.dtype | str): Float dtype of the scores, for example
            ``torch.float32`` or ``"float64"``. Counts stay exact: weighted counts switch
            to ``float64`` when ``dtype`` cannot represent them exactly. MPS has no
            ``float64``. Stored as a ``torch.dtype``. Default: `torch.float32`.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset
        >>> from hgp_lib.evaluation.torch import TorchBackend
        >>> from hgp_lib.rules import Literal, Or
        >>> backend = TorchBackend()
        >>> backend.device
        device(type='cpu')
        >>> data = np.array([[True, False], [False, True], [False, False]])
        >>> evaluator = backend.bind(Dataset(data, np.array([True, True, False])))
        >>> scores = evaluator.score([Literal(value=0), Or([Literal(value=0), Literal(value=1)])])
        >>> backend.to_numpy(scores).tolist()
        [0.6666666865348816, 1.0]
        >>> evaluator.confusion_matrix(Literal(value=1))
        (1, 0, 1, 1)
    """

    device: torch.device | str | None = None
    evaluation: Literal["block", "inplace", "fused"] = "fused"
    batched: bool = False
    batch_size: int | None = None
    dtype: torch.dtype | str = torch.float32

    def __post_init__(self):
        if self.device is not None and not isinstance(self.device, (str, torch.device)):
            raise TypeError(
                f"device must be a torch.device, a str or None, is {type(self.device)}"
            )
        device = torch.device("cpu" if self.device is None else self.device)
        object.__setattr__(self, "device", device)
        check_evaluation(self.evaluation)
        validate_batching(self.batched, self.batch_size)
        dtype = (
            getattr(torch, self.dtype, None)
            if isinstance(self.dtype, str)
            else self.dtype
        )
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"dtype must be a float torch.dtype, is {self.dtype!r}")
        if device.type == "mps" and dtype == torch.float64:
            raise ValueError("MPS has no float64; use dtype=torch.float32")
        object.__setattr__(self, "dtype", dtype)

    def _bind(
        self, dataset: Dataset | TorchDataset, score_fn: ScoreFn, merge_rows: bool
    ) -> "TorchEvaluator":
        if isinstance(dataset, TorchDataset):
            dataset = self._on_device(dataset)
        else:
            dataset = TorchDataset.from_numpy(dataset, self.device)
        return TorchEvaluator(self, dataset, score_fn, merge_rows)

    def _on_device(self, dataset: TorchDataset) -> TorchDataset:
        """The dataset on this backend's device, with column-major data."""
        data, labels, weights = dataset
        if data.device != self.device or data.stride(0) != 1:
            data = data.to(self.device).T.contiguous().T
        return TorchDataset(
            data,
            labels.to(self.device),
            None if weights is None else weights.to(self.device),
        )

    def predict(self, rule: Rule, data: Tensor) -> Tensor:
        return EVALUATIONS[self.evaluation](rule, data)

    def from_numpy(self, data: ndarray) -> Tensor:
        if self.device.type == "cpu":
            # Shares memory with ``data``, which is only read.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "The given NumPy array is not writable"
                )
                return torch.from_numpy(data)
        return column_major(data, self.device)

    def to_numpy(self, values: Any) -> ndarray:
        if isinstance(values, Tensor):
            return values.cpu().numpy()
        return np.asarray(values)


def _as_scores(values: list, dtype: torch.dtype, device: torch.device) -> Tensor:
    """One tensor from the scores of the rules: tensors, or Python floats."""
    if not values:
        return torch.empty(0, dtype=dtype, device=device)
    if isinstance(values[0], Tensor):
        return torch.stack(values).to(device=device, dtype=dtype)
    return torch.tensor(values, dtype=dtype, device=device)


class TorchEvaluator(Evaluator):
    """`Evaluator` of the `TorchBackend`. Created by `TorchBackend.bind`."""

    def __init__(
        self,
        backend: TorchBackend,
        dataset: TorchDataset,
        score_fn: ScoreFn,
        merge_rows: bool,
    ):
        super().__init__(backend, dataset, score_fn, merge_rows)
        self._evaluate = EVALUATIONS[backend.evaluation]
        self._dtype = backend.dtype
        self._device = backend.device
        self._kernel, self._confusion = bind_kernels(dataset, score_fn, backend.dtype)
        # Kernels without a batched method take one prediction per call.
        self._batched = backend.batched and hasattr(self._kernel, "batched")
        self._batch_size = backend.batch_size

    def score(self, rules: Sequence[Rule]) -> Tensor:
        if self._batched and rules:
            return self._score_batched(rules)
        data = self.dataset.data
        evaluate_rule = self._evaluate
        kernel = self._kernel
        values = [kernel(evaluate_rule(rule, data)) for rule in rules]
        return _as_scores(values, self._dtype, self._device)

    def _score_batched(self, rules: Sequence[Rule]) -> Tensor:
        data = self.dataset.data
        evaluate_rule = self._evaluate
        score_batch = self._kernel.batched
        size = self._batch_size or len(rules)
        # One row per rule, reused by every batch.
        block = torch.empty(
            (min(size, len(rules)), data.shape[0]), dtype=torch.bool, device=data.device
        )
        scores = []
        for start in range(0, len(rules), size):
            batch = rules[start : start + size]
            predictions = block[: len(batch)]
            for row, rule in zip(predictions, batch):
                row.copy_(evaluate_rule(rule, data))
            scores.append(
                torch.as_tensor(
                    score_batch(predictions), dtype=self._dtype, device=self._device
                )
            )
        return torch.cat(scores)

    def confusion_matrix(self, rule: Rule) -> tuple[int, int, int, int]:
        return self._confusion(self._evaluate(rule, self.dataset.data))
