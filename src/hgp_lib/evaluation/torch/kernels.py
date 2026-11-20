"""
PyTorch scoring kernels bound to one dataset: the kernels of
`hgp_lib.evaluation.numpy.kernels`, with the torch API.

Kernels return their scores as tensors on the backend's device, in the backend's
``dtype``. Nothing is copied to the host while scoring.
"""

import math

import torch
from torch import Tensor, count_nonzero

from ..scorers import ScoreFn, fast_accuracy_score, fast_f1_score
from .dataset import TorchDataset


def count_dtype(
    dtype: torch.dtype, total_weight: int, device: torch.device
) -> torch.dtype:
    """
    The float dtype that represents every count exactly: ``dtype``, or ``float64``.

    See `hgp_lib.evaluation.numpy.kernels.count_dtype`.

    Raises:
        ValueError: If the counts need ``float64`` on MPS, which has no ``float64``.

    Examples:
        >>> import torch
        >>> from hgp_lib.evaluation.torch.kernels import count_dtype
        >>> count_dtype(torch.float32, 1000, torch.device("cpu"))
        torch.float32
        >>> count_dtype(torch.float32, 2**24, torch.device("cpu"))
        torch.float64
    """
    mantissa_bits = round(-math.log2(torch.finfo(dtype).eps))
    if 2 * total_weight <= 2 ** (mantissa_bits + 1):
        return dtype
    if device.type == "mps":
        raise ValueError(
            f"Scores of {total_weight} original rows need float64 counts to stay "
            f"exact with dtype={dtype}, and MPS has no float64. Use a CPU or CUDA "
            "device for this dataset."
        )
    return torch.float64


def count_matrix(dataset: TorchDataset, dtype: torch.dtype) -> Tensor:
    """
    The ``(n, 2)`` count matrix ``[w, w * y]`` of a weighted dataset, in ``dtype``.

    ``y_pred @ matrix`` is ``[predicted, tp]``.
    """
    weights = dataset.sample_weight
    return torch.stack((weights, weights * dataset.labels), dim=1).to(dtype)


class _NoPositives:
    """F1 when no label is positive: 1.0 if nothing is predicted positive, else 0.0."""

    __slots__ = ("dtype",)

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def __call__(self, y_pred: Tensor) -> Tensor:
        return (~y_pred.any()).to(self.dtype)

    def batched(self, y_preds: Tensor) -> Tensor:
        return (~y_preds.any(dim=1)).to(self.dtype)


class _F1:
    __slots__ = ("labels", "positives", "count_dtype", "dtype")

    def __init__(
        self,
        labels: Tensor,
        positives: int,
        count_dtype: torch.dtype,
        dtype: torch.dtype,
    ):
        self.labels = labels
        self.positives = positives
        self.count_dtype = count_dtype
        self.dtype = dtype

    def __call__(self, y_pred: Tensor) -> Tensor:
        tp = count_nonzero(y_pred & self.labels).to(self.count_dtype)
        scores = 2 * tp / (count_nonzero(y_pred) + self.positives)
        return scores.to(self.dtype)

    def batched(self, y_preds: Tensor) -> Tensor:
        tp = count_nonzero(y_preds & self.labels, dim=1).to(self.count_dtype)
        scores = 2 * tp / (count_nonzero(y_preds, dim=1) + self.positives)
        return scores.to(self.dtype)


class _F1Weighted:
    __slots__ = ("matrix", "positives", "dtype")

    def __init__(self, matrix: Tensor, positives: int, dtype: torch.dtype):
        self.matrix = matrix
        self.positives = positives
        self.dtype = dtype

    def __call__(self, y_pred: Tensor) -> Tensor:
        predicted, tp = y_pred.to(self.matrix.dtype) @ self.matrix
        return (2 * tp / (predicted + self.positives)).to(self.dtype)

    def batched(self, y_preds: Tensor) -> Tensor:
        counts = y_preds.to(self.matrix.dtype) @ self.matrix
        denominator = counts[:, 0]
        scores = counts[:, 1]
        denominator += self.positives
        scores *= 2
        scores /= denominator
        return scores.to(self.dtype)


class _Accuracy:
    __slots__ = ("labels", "total", "count_dtype", "dtype")

    def __init__(
        self,
        labels: Tensor,
        total: int,
        count_dtype: torch.dtype,
        dtype: torch.dtype,
    ):
        self.labels = labels
        self.total = total
        self.count_dtype = count_dtype
        self.dtype = dtype

    def __call__(self, y_pred: Tensor) -> Tensor:
        correct = count_nonzero(y_pred == self.labels).to(self.count_dtype)
        return (correct / self.total).to(self.dtype)

    def batched(self, y_preds: Tensor) -> Tensor:
        correct = count_nonzero(y_preds == self.labels, dim=1).to(self.count_dtype)
        return (correct / self.total).to(self.dtype)


class _AccuracyWeighted:
    __slots__ = ("matrix", "negatives", "total", "dtype")

    def __init__(self, matrix: Tensor, positives: int, total: int, dtype: torch.dtype):
        self.matrix = matrix
        self.negatives = total - positives
        self.total = total
        self.dtype = dtype

    def __call__(self, y_pred: Tensor) -> Tensor:
        # Correct rows are the true positives plus the true negatives.
        predicted, tp = y_pred.to(self.matrix.dtype) @ self.matrix
        return ((self.negatives - predicted + 2 * tp) / self.total).to(self.dtype)

    def batched(self, y_preds: Tensor) -> Tensor:
        counts = y_preds.to(self.matrix.dtype) @ self.matrix
        predicted = counts[:, 0]
        scores = counts[:, 1]
        scores *= 2
        scores -= predicted
        scores += self.negatives
        scores /= self.total
        return scores.to(self.dtype)


class _Custom:
    __slots__ = ("fn", "labels")

    def __init__(self, fn: ScoreFn, labels: Tensor):
        self.fn = fn
        self.labels = labels

    def __call__(self, y_pred: Tensor):
        return self.fn(self.labels, y_pred)


class _CustomBatched(_Custom):
    __slots__ = ()

    def batched(self, y_preds: Tensor):
        return self.fn.batched(self.labels, y_preds)


class _CustomWeighted:
    __slots__ = ("fn", "labels", "weights")

    def __init__(self, fn: ScoreFn, labels: Tensor, weights: Tensor):
        self.fn = fn
        self.labels = labels
        self.weights = weights

    def __call__(self, y_pred: Tensor):
        return self.fn(self.labels, y_pred, sample_weight=self.weights)


class _CustomWeightedBatched(_CustomWeighted):
    __slots__ = ()

    def batched(self, y_preds: Tensor):
        return self.fn.batched(self.labels, y_preds, sample_weight=self.weights)


class ConfusionCounts:
    """Bound confusion matrix: ``counts(y_pred) -> (tp, fp, fn, tn)`` over the original rows."""

    __slots__ = ("labels", "matrix", "positives", "total")

    def __init__(
        self, labels: Tensor, matrix: Tensor | None, positives: int, total: int
    ):
        self.labels = labels
        self.matrix = matrix
        self.positives = positives
        self.total = total

    def __call__(self, y_pred: Tensor) -> tuple[int, int, int, int]:
        if self.matrix is None:
            counts = torch.stack(
                (count_nonzero(y_pred), count_nonzero(y_pred & self.labels))
            )
        else:
            counts = y_pred.to(self.matrix.dtype) @ self.matrix
        predicted, tp = (int(count) for count in counts.tolist())
        return (
            tp,
            predicted - tp,
            self.positives - tp,
            self.total - self.positives - predicted + tp,
        )


def bind_kernels(dataset: TorchDataset, score_fn: ScoreFn, dtype: torch.dtype):
    """
    Bind a scoring function to a dataset: a scoring kernel and a confusion matrix.

    `fast_f1_score` and `fast_accuracy_score` get dedicated kernels. Any other scoring
    function receives tensors: the dataset's labels, the prediction, and
    ``sample_weight`` only when the dataset is weighted. Its kernel has a ``batched``
    method when the function has one.

    Args:
        dataset (TorchDataset): The (possibly merged) dataset.
        score_fn (ScoreFn): The scoring function.
        dtype (torch.dtype): The score dtype of the backend.

    Returns:
        tuple: ``(kernel, confusion)``.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from hgp_lib.evaluation import Dataset, fast_f1_score
        >>> from hgp_lib.evaluation.torch import TorchDataset
        >>> from hgp_lib.evaluation.torch.kernels import bind_kernels
        >>> rows = Dataset(np.zeros((3, 1), dtype=bool), np.array([True, False, True]), np.array([2, 1, 1]))
        >>> dataset = TorchDataset.from_numpy(rows, torch.device("cpu"))
        >>> kernel, confusion = bind_kernels(dataset, fast_f1_score, torch.float64)
        >>> float(kernel(torch.tensor([True, False, False])))
        0.8
        >>> confusion(torch.tensor([True, False, False]))
        (2, 0, 1, 1)
    """
    labels = dataset.labels
    weights = dataset.sample_weight
    if weights is None:
        positives = int(count_nonzero(labels))
        total = len(labels)
    else:
        positives = int(weights[labels].sum())
        total = int(weights.sum())
    counts = count_dtype(dtype, total, labels.device)
    matrix = None if weights is None else count_matrix(dataset, counts)
    confusion = ConfusionCounts(labels, matrix, positives, total)

    if score_fn is fast_f1_score:
        if not positives:
            kernel = _NoPositives(dtype)
        elif matrix is None:
            kernel = _F1(labels, positives, counts, dtype)
        else:
            kernel = _F1Weighted(matrix, positives, dtype)
    elif score_fn is fast_accuracy_score:
        if matrix is None:
            kernel = _Accuracy(labels, total, counts, dtype)
        else:
            kernel = _AccuracyWeighted(matrix, positives, total, dtype)
    else:
        batched = callable(getattr(score_fn, "batched", None))
        if weights is None:
            kernel = (_CustomBatched if batched else _Custom)(score_fn, labels)
        else:
            kernel_class = _CustomWeightedBatched if batched else _CustomWeighted
            kernel = kernel_class(score_fn, labels, weights)
    return kernel, confusion
