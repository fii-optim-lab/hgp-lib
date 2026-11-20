"""
NumPy scoring kernels bound to one dataset.

Each kernel is a small callable that takes only the predictions: labels, weights and
constants such as the number of positive labels are computed once at bind time, and the
branch for datasets without positive labels is chosen once, not per call. Kernels with a
``batched`` method also score a 2-D stack of predictions, one row per rule: the built-in
kernels, and custom scoring functions that have a ``batched`` method.

Counts are exact integers in the unweighted and the weighted paths. Weighted counts come
from one product with the ``(n, 2)`` count matrix ``[w, w * y]``, in a float dtype that
represents every count exactly (see `count_dtype`).
"""

import numpy as np
from numpy import count_nonzero, dot, ndarray

from ..dataset import Dataset
from ..scorers import ScoreFn, fast_accuracy_score, fast_f1_score


def count_dtype(dtype: np.dtype, total_weight: int) -> np.dtype:
    """
    The float dtype for exact weighted counts: ``dtype``, or ``float64`` if needed.

    Scores need ``2 * tp`` and ``predicted + positives``, which reach twice the total
    weight. A float dtype with ``m`` mantissa bits represents every integer up to
    ``2**(m + 1)`` exactly, for example 16,777,216 for ``float32``.

    Args:
        dtype (np.dtype): The score dtype of the backend.
        total_weight (int): The number of original rows.

    Returns:
        np.dtype: ``dtype`` when its counts are exact, otherwise ``float64``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.numpy.kernels import count_dtype
        >>> count_dtype(np.dtype(np.float32), 1000)
        dtype('float32')
        >>> count_dtype(np.dtype(np.float32), 2**24)
        dtype('float64')
    """
    if 2 * total_weight <= 2 ** (np.finfo(dtype).nmant + 1):
        return dtype
    return np.dtype(np.float64)


def count_matrix(dataset: Dataset, dtype: np.dtype) -> ndarray:
    """
    The ``(n, 2)`` count matrix ``[w, w * y]`` of a weighted dataset, in column-major order.

    ``y_pred @ matrix`` is ``[predicted, tp]``: the weights of the predicted positive
    rows and of the true positive rows.

    Args:
        dataset (Dataset): A dataset with sample weights.
        dtype (np.dtype): The score dtype of the backend.

    Returns:
        ndarray: The count matrix, in `count_dtype` for the dataset's total weight.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset
        >>> from hgp_lib.evaluation.numpy.kernels import count_matrix
        >>> dataset = Dataset(np.zeros((2, 1), dtype=bool), np.array([True, False]), np.array([3, 2]))
        >>> count_matrix(dataset, np.dtype(np.float32)).tolist()
        [[3.0, 3.0], [2.0, 0.0]]
    """
    weights = dataset.sample_weight
    matrix = np.empty(
        (len(weights), 2), dtype=count_dtype(dtype, int(weights.sum())), order="F"
    )
    matrix[:, 0] = weights
    matrix[:, 1] = weights * dataset.labels
    return matrix


class _NoPositives:
    """F1 when no label is positive: 1.0 if nothing is predicted positive, else 0.0."""

    __slots__ = ()

    def __call__(self, y_pred: ndarray) -> float:
        # Weights are positive counts, so any positive prediction has positive weight.
        return 0.0 if y_pred.any() else 1.0

    def batched(self, y_preds: ndarray) -> ndarray:
        return np.where(y_preds.any(axis=1), 0.0, 1.0)


class _F1:
    __slots__ = ("labels", "positives")

    def __init__(self, labels: ndarray, positives: int):
        self.labels = labels
        self.positives = positives

    def __call__(self, y_pred: ndarray) -> float:
        return (
            2
            * count_nonzero(y_pred & self.labels)
            / (count_nonzero(y_pred) + self.positives)
        )

    def batched(self, y_preds: ndarray) -> ndarray:
        return (
            2
            * count_nonzero(y_preds & self.labels, axis=1)
            / (count_nonzero(y_preds, axis=1) + self.positives)
        )


class _F1Weighted:
    __slots__ = ("matrix", "positives")

    def __init__(self, matrix: ndarray, positives: int):
        self.matrix = matrix
        self.positives = positives

    def __call__(self, y_pred: ndarray) -> float:
        predicted, tp = dot(y_pred, self.matrix)
        return 2 * tp / (predicted + self.positives)

    def batched(self, y_preds: ndarray) -> ndarray:
        counts = y_preds @ self.matrix
        denominator = counts[:, 0]
        scores = counts[:, 1]
        denominator += self.positives
        scores *= 2
        scores /= denominator
        return scores


class _Accuracy:
    __slots__ = ("labels", "total")

    def __init__(self, labels: ndarray, total: int):
        self.labels = labels
        self.total = total

    def __call__(self, y_pred: ndarray) -> float:
        return count_nonzero(y_pred == self.labels) / self.total

    def batched(self, y_preds: ndarray) -> ndarray:
        return count_nonzero(y_preds == self.labels, axis=1) / self.total


class _AccuracyWeighted:
    __slots__ = ("matrix", "negatives", "total")

    def __init__(self, matrix: ndarray, positives: int, total: int):
        self.matrix = matrix
        self.negatives = total - positives
        self.total = total

    def __call__(self, y_pred: ndarray) -> float:
        # Correct rows are the true positives plus the true negatives.
        predicted, tp = dot(y_pred, self.matrix)
        return (self.negatives - predicted + 2 * tp) / self.total

    def batched(self, y_preds: ndarray) -> ndarray:
        counts = y_preds @ self.matrix
        predicted = counts[:, 0]
        scores = counts[:, 1]
        scores *= 2
        scores -= predicted
        scores += self.negatives
        scores /= self.total
        return scores


class _Custom:
    __slots__ = ("fn", "labels")

    def __init__(self, fn: ScoreFn, labels: ndarray):
        self.fn = fn
        self.labels = labels

    def __call__(self, y_pred: ndarray) -> float:
        return self.fn(self.labels, y_pred)


class _CustomBatched(_Custom):
    __slots__ = ()

    def batched(self, y_preds: ndarray) -> ndarray:
        return self.fn.batched(self.labels, y_preds)


class _CustomWeighted:
    __slots__ = ("fn", "labels", "weights")

    def __init__(self, fn: ScoreFn, labels: ndarray, weights: ndarray):
        self.fn = fn
        self.labels = labels
        self.weights = weights

    def __call__(self, y_pred: ndarray) -> float:
        return self.fn(self.labels, y_pred, sample_weight=self.weights)


class _CustomWeightedBatched(_CustomWeighted):
    __slots__ = ()

    def batched(self, y_preds: ndarray) -> ndarray:
        return self.fn.batched(self.labels, y_preds, sample_weight=self.weights)


class ConfusionCounts:
    """
    Bound confusion matrix: ``counts(y_pred) -> (tp, fp, fn, tn)`` over the original rows.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset
        >>> from hgp_lib.evaluation.numpy.kernels import ConfusionCounts, count_matrix
        >>> dataset = Dataset(np.zeros((4, 1), dtype=bool), np.array([True, False, True, False]), np.array([2, 3, 1, 1]))
        >>> counts = ConfusionCounts(dataset.labels, count_matrix(dataset, np.dtype(np.float32)), 3, 7)
        >>> counts(np.array([True, True, False, False]))
        (2, 3, 1, 1)
    """

    __slots__ = ("labels", "matrix", "positives", "total")

    def __init__(
        self, labels: ndarray, matrix: ndarray | None, positives: int, total: int
    ):
        self.labels = labels
        self.matrix = matrix
        self.positives = positives
        self.total = total

    def __call__(self, y_pred: ndarray) -> tuple[int, int, int, int]:
        if self.matrix is None:
            tp = int(count_nonzero(y_pred & self.labels))
            predicted = int(count_nonzero(y_pred))
        else:
            predicted, tp = (int(count) for count in dot(y_pred, self.matrix))
        return (
            tp,
            predicted - tp,
            self.positives - tp,
            self.total - self.positives - predicted + tp,
        )


def bind_kernels(dataset: Dataset, score_fn: ScoreFn, dtype: np.dtype):
    """
    Bind a scoring function to a dataset: a scoring kernel and a confusion matrix.

    `fast_f1_score` and `fast_accuracy_score` get dedicated kernels. Any other scoring
    function is called with the dataset's labels, and with ``sample_weight`` only when
    the dataset is weighted. Its kernel has a ``batched`` method when the function has
    one.

    Args:
        dataset (Dataset): The (possibly merged) dataset.
        score_fn (ScoreFn): The scoring function.
        dtype (np.dtype): The score dtype of the backend.

    Returns:
        tuple: ``(kernel, confusion)``, where ``kernel(y_pred)`` scores one prediction
            and ``confusion(y_pred)`` returns ``(tp, fp, fn, tn)``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset, fast_f1_score
        >>> from hgp_lib.evaluation.numpy.kernels import bind_kernels
        >>> dataset = Dataset(np.zeros((3, 1), dtype=bool), np.array([True, False, True]), np.array([2, 1, 1]))
        >>> kernel, confusion = bind_kernels(dataset, fast_f1_score, np.dtype(np.float64))
        >>> float(kernel(np.array([True, False, False])))
        0.8
        >>> kernel.batched(np.array([[True, False, False], [True, True, True]])).tolist()
        [0.8, 0.8571428571428571]
        >>> confusion(np.array([True, False, False]))
        (2, 0, 1, 1)
    """
    labels = dataset.labels
    weights = dataset.sample_weight
    if weights is None:
        matrix = None
        positives = int(count_nonzero(labels))
        total = len(labels)
    else:
        matrix = count_matrix(dataset, dtype)
        positives = int(weights[labels].sum())
        total = int(weights.sum())
    confusion = ConfusionCounts(labels, matrix, positives, total)

    if score_fn is fast_f1_score:
        if not positives:
            kernel = _NoPositives()
        elif matrix is None:
            kernel = _F1(labels, positives)
        else:
            kernel = _F1Weighted(matrix, positives)
    elif score_fn is fast_accuracy_score:
        if matrix is None:
            kernel = _Accuracy(labels, total)
        else:
            kernel = _AccuracyWeighted(matrix, positives, total)
    else:
        batched = callable(getattr(score_fn, "batched", None))
        if weights is None:
            kernel = (_CustomBatched if batched else _Custom)(score_fn, labels)
        else:
            kernel_class = _CustomWeightedBatched if batched else _CustomWeighted
            kernel = kernel_class(score_fn, labels, weights)
    return kernel, confusion
