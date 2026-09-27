import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from ...rules import Rule
from ...utils.validation import validate_batching
from ..backend import EvaluationBackend, Evaluator
from ..dataset import Dataset
from ..scorers import Scorer, fast_accuracy_score, fast_f1_score
from .predict import evaluate


def _f1_from_counts(
    tp: ndarray, predicted: ndarray, positives: int, total: int
) -> ndarray:
    if not positives:
        # 1.0 if nothing is predicted positive, else 0.0, as fast_f1_score.
        return np.where(predicted > 0, 0.0, 1.0)
    return 2 * tp / (predicted + positives)


def _accuracy_from_counts(
    tp: ndarray, predicted: ndarray, positives: int, total: int
) -> ndarray:
    # Correct rows are the true positives plus the true negatives.
    return (total - positives - predicted + 2 * tp) / total


def _from_counts(scorer: Scorer):
    """The function turning counts into scores, or ``None`` for a custom scorer."""
    if scorer.fn is fast_f1_score:
        return _f1_from_counts
    if scorer.fn is fast_accuracy_score:
        return _accuracy_from_counts
    return None


@dataclass(frozen=True, kw_only=True)
class TorchBackend(EvaluationBackend):
    """
    Evaluate rules with PyTorch, on the CPU or on an accelerator.

    Needs PyTorch (``pip install "hgp-lib[torch]"``). Gives the same scores as
    `NumpyBackend`: the built-in scorers count true positives and predicted positives as
    exact integers on the device, and only the counts are copied back. Custom scorers
    receive each prediction as a NumPy array, which copies it from the device.

    A rule costs one tensor operation per literal, and every operation has a fixed launch
    cost, so an accelerator only pays off on large data. On an Apple M3 GPU (``"mps"``),
    scoring 100 random rules was 1.6x faster than `NumpyBackend` on 1 million rows, and
    4.6x slower on 100 thousand rows. In the repository's backend benchmarks (up to 37
    thousand rows), `NumpyBackend` was faster in every scenario, so on the CPU use
    `NumpyBackend`.

    Attributes:
        device (torch.device | str | None): Where the bound data is stored and rules are
            evaluated, for example ``"cuda"``, ``"cuda:1"`` or ``"mps"``. ``None`` means
            ``"cpu"``. Stored as a ``torch.device``. Default: `None`.
        batched (bool): Stack the predictions of ``batch_size`` rules and count them
            with a few operations per batch, instead of two per rule. Only applies to
            the built-in scorers; custom scorers take one prediction per call. In the
            backend benchmarks it was up to 17% faster on MPS, and between 12% faster
            and 22% slower on the CPU. Default: `False`.
        batch_size (int | None): Number of rules per batch when ``batched`` is
            ``True``. ``None`` scores all rules in one batch. A batch needs a boolean
            block of ``batch_size`` times the number of bound rows on the device, plus
            an ``int64`` block of the same shape for weighted data. Setting it requires
            ``batched=True``. Default: `None`.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset
        >>> from hgp_lib.evaluation.scorers import resolve_scorer
        >>> from hgp_lib.evaluation.torch import TorchBackend
        >>> from hgp_lib.rules import Literal, Or
        >>> backend = TorchBackend()
        >>> backend.device
        device(type='cpu')
        >>> data = np.array([[True, False], [False, True], [False, False]])
        >>> evaluator = backend.bind(Dataset(data, np.array([1, 1, 0])), resolve_scorer())
        >>> evaluator.score([Literal(value=0), Or([Literal(value=0), Literal(value=1)])]).tolist()
        [0.6666666666666666, 1.0]
        >>> evaluator.confusion_matrix(Literal(value=1))
        (1, 0, 1, 1)
    """

    device: torch.device | str | None = None
    batched: bool = False
    batch_size: int | None = None

    def __post_init__(self):
        if self.device is not None and not isinstance(self.device, (str, torch.device)):
            raise TypeError(
                f"device must be a torch.device, a str or None, is {type(self.device)}"
            )
        device = torch.device("cpu" if self.device is None else self.device)
        object.__setattr__(self, "device", device)
        validate_batching(self.batched, self.batch_size)

    def _bind(self, dataset: Dataset, scorer: Scorer) -> "TorchEvaluator":
        return TorchEvaluator(
            dataset, scorer, self.device, self.batched, self.batch_size
        )

    def predict(self, rule: Rule, data: ndarray) -> ndarray:
        if self.device.type == "cpu":
            # Shares memory with ``data``, which is only read.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "The given NumPy array is not writable"
                )
                columns = torch.from_numpy(data).T
        else:
            columns = torch.tensor(np.ascontiguousarray(data.T), device=self.device)
        return evaluate(rule, columns).cpu().numpy()


class TorchEvaluator(Evaluator):
    """
    `Evaluator` of the `TorchBackend`. Created by `TorchBackend.bind`.

    ``dataset`` stays a NumPy `Dataset`, for strategies that read or sample it. The
    evaluator keeps a copy of the features, labels and weights on the device.
    """

    def __init__(
        self,
        dataset: Dataset,
        scorer: Scorer,
        device: torch.device,
        batched: bool,
        batch_size: int | None,
    ):
        super().__init__(dataset, scorer)
        labels = dataset.labels.astype(bool, copy=False)
        weights = dataset.sample_weight
        # One contiguous row per feature, so each literal reads contiguous memory.
        self._columns = torch.tensor(
            np.ascontiguousarray(dataset.data.T), device=device
        )
        # ``(count_rows * prediction).sum(-1)`` is ``[true positives, predicted
        # positives]``: two tensor operations per rule.
        if weights is None:
            count_rows = np.stack((labels, np.ones_like(labels)))
            self._positives = int(np.count_nonzero(labels))
            self._total = len(labels)
        else:
            count_rows = np.stack((weights * labels, weights))
            self._positives = int(count_rows[0].sum())
            self._total = int(weights.sum())
        self._count_rows = torch.tensor(count_rows, device=device)
        self._from_counts = _from_counts(scorer)
        self._batched = batched and self._from_counts is not None
        self._batch_size = batch_size

    def _counts(self, prediction: Tensor) -> Tensor:
        """``[true positives, predicted positives]`` of one prediction, on the device."""
        return (self._count_rows * prediction).sum(-1)

    def score(self, rules: Sequence[Rule]) -> ndarray:
        if not rules:
            return np.empty(0)
        if self._from_counts is None:
            return self._score_custom(rules)
        if self._batched:
            counts = self._counts_batched(rules)
        else:
            columns = self._columns
            counts = torch.stack(
                [self._counts(evaluate(rule, columns)) for rule in rules]
            )
        # One copy from the device for the whole population.
        tp, predicted = counts.cpu().numpy().T
        return self._from_counts(tp, predicted, self._positives, self._total)

    def _counts_batched(self, rules: Sequence[Rule]) -> Tensor:
        columns = self._columns
        positive_row, predicted_row = self._count_rows
        size = self._batch_size or len(rules)
        # One row per rule, reused by every batch.
        block = torch.empty(
            (min(size, len(rules)), columns.shape[1]),
            dtype=torch.bool,
            device=columns.device,
        )
        counts = []
        for start in range(0, len(rules), size):
            batch = rules[start : start + size]
            predictions = block[: len(batch)]
            for row, rule in zip(predictions, batch):
                row.copy_(evaluate(rule, columns))
            tp = (predictions * positive_row).sum(-1)
            predicted = (predictions * predicted_row).sum(-1)
            counts.append(torch.stack((tp, predicted), dim=1))
        return torch.cat(counts)

    def _score_custom(self, rules: Sequence[Rule]) -> ndarray:
        labels, weights = self.dataset.labels, self.dataset.sample_weight
        scorer, columns = self.scorer, self._columns
        return np.fromiter(
            (
                scorer(labels, evaluate(rule, columns).cpu().numpy(), weights)
                for rule in rules
            ),
            dtype=np.float64,
            count=len(rules),
        )

    def confusion_matrix(self, rule: Rule) -> tuple[int, int, int, int]:
        tp, predicted = self._counts(evaluate(rule, self._columns)).tolist()
        positives, total = self._positives, self._total
        return (
            tp,
            predicted - tp,
            positives - tp,
            total - positives - predicted + tp,
        )
