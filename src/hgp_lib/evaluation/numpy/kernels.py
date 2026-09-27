"""
NumPy scoring kernels bound to one dataset.

Each kernel is a small callable that takes only the predictions: labels, weights and
constants such as the number of positive labels are computed once at bind time, and the
branch for datasets without positive labels is chosen once, not per call. The built-in
kernels also have a ``batched`` method, which scores a stack of predictions (one row per
rule) against the same labels.

Counts are exact integers in both the weighted and unweighted paths, so a merged
dataset gives bit-for-bit the same scores as the original rows, one rule at a time or
batched.
"""

import numpy as np
from numpy import count_nonzero, dot, ndarray

from ..dataset import Dataset
from ..scorers import Scorer, fast_accuracy_score, fast_f1_score

# TODO: Benchmark float32 weights for weighted counting (exact while the total weight
#  stays below 2**24). Not planned for 2.0.0.


def _count_matrix(weights: ndarray, positive_weights: ndarray) -> ndarray:
    """
    The ``[w, w * y]`` columns as ``float64``.

    ``predictions @ matrix`` gives the predicted-positive and true-positive weights of
    every rule in one matrix product. ``float64`` keeps integer sums exact below 2**53.
    """
    return np.column_stack((weights, positive_weights)).astype(np.float64)


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
    __slots__ = ("weights", "positive_weights", "positives", "matrix")

    def __init__(self, weights: ndarray, positive_weights: ndarray, positives: int):
        self.weights = weights
        self.positive_weights = positive_weights
        self.positives = positives
        self.matrix = _count_matrix(weights, positive_weights)

    def __call__(self, y_pred: ndarray) -> float:
        return (
            2
            * dot(y_pred, self.positive_weights)
            / (dot(y_pred, self.weights) + self.positives)
        )

    def batched(self, y_preds: ndarray) -> ndarray:
        predicted, true_positives = (y_preds @ self.matrix).T
        return 2 * true_positives / (predicted + self.positives)


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
    __slots__ = ("labels", "weights", "positives", "total", "matrix")

    def __init__(
        self,
        labels: ndarray,
        weights: ndarray,
        positive_weights: ndarray,
        positives: int,
        total: int,
    ):
        self.labels = labels
        self.weights = weights
        self.positives = positives
        self.total = total
        self.matrix = _count_matrix(weights, positive_weights)

    def __call__(self, y_pred: ndarray) -> float:
        return dot(y_pred == self.labels, self.weights) / self.total

    def batched(self, y_preds: ndarray) -> ndarray:
        predicted, true_positives = (y_preds @ self.matrix).T
        # Correct rows are the true positives plus the true negatives.
        correct = self.total - self.positives - predicted + 2 * true_positives
        return correct / self.total


class _Custom:
    __slots__ = ("fn", "labels")

    def __init__(self, fn, labels: ndarray):
        self.fn = fn
        self.labels = labels

    def __call__(self, y_pred: ndarray) -> float:
        return self.fn(self.labels, y_pred)


class _CustomWeighted:
    __slots__ = ("fn", "labels", "weights")

    def __init__(self, fn, labels: ndarray, weights: ndarray):
        self.fn = fn
        self.labels = labels
        self.weights = weights

    def __call__(self, y_pred: ndarray) -> float:
        return self.fn(self.labels, y_pred, sample_weight=self.weights)


def _bool_labels(dataset: Dataset) -> ndarray:
    return dataset.labels.astype(bool, copy=False)


def bind_kernel(dataset: Dataset, scorer: Scorer):
    """
    Bind a scorer to a dataset, returning a callable ``kernel(y_pred) -> float``.

    `fast_f1_score` and `fast_accuracy_score` get dedicated kernels, which also have
    ``kernel.batched(y_preds) -> ndarray`` for a 2D stack of predictions, one row per
    rule. Any other scorer is called with the dataset's own labels, and with
    ``sample_weight`` only when the dataset is weighted. It has no ``batched`` method.

    Args:
        dataset (Dataset): The (possibly merged) dataset.
        scorer (Scorer): The resolved scorer.

    Returns:
        Callable[[ndarray], float]: The bound kernel.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset, Scorer, fast_f1_score
        >>> from hgp_lib.evaluation.numpy.kernels import bind_kernel
        >>> dataset = Dataset(np.zeros((3, 1), dtype=bool), np.array([1, 0, 1]), np.array([2, 1, 1]))
        >>> kernel = bind_kernel(dataset, Scorer(fast_f1_score, merge_rows=True))
        >>> float(kernel(np.array([True, False, False])))
        0.8
        >>> kernel.batched(np.array([[True, False, False], [True, True, True]])).tolist()
        [0.8, 0.8571428571428571]
    """
    fn = scorer.fn
    weights = dataset.sample_weight
    if fn is fast_f1_score or fn is fast_accuracy_score:
        labels = _bool_labels(dataset)
        if weights is None:
            if fn is fast_accuracy_score:
                return _Accuracy(labels, len(labels))
            positives = int(count_nonzero(labels))
            return _F1(labels, positives) if positives else _NoPositives()
        positive_weights = weights * labels
        positives = int(positive_weights.sum())
        if fn is fast_accuracy_score:
            return _AccuracyWeighted(
                labels, weights, positive_weights, positives, int(weights.sum())
            )
        if not positives:
            return _NoPositives()
        return _F1Weighted(weights, positive_weights, positives)
    if weights is None:
        return _Custom(fn, dataset.labels)
    return _CustomWeighted(fn, dataset.labels, weights)


class ConfusionCounts:
    """
    Bound confusion matrix: ``counts(y_pred) -> (tp, fp, fn, tn)`` over the original rows.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset
        >>> from hgp_lib.evaluation.numpy.kernels import ConfusionCounts
        >>> dataset = Dataset(np.zeros((4, 1), dtype=bool), np.array([1, 0, 1, 0]), np.array([2, 3, 1, 1]))
        >>> ConfusionCounts(dataset)(np.array([True, True, False, False]))
        (2, 3, 1, 1)
    """

    __slots__ = ("labels", "weights", "positive_weights", "positives", "total")

    def __init__(self, dataset: Dataset):
        self.labels = _bool_labels(dataset)
        self.weights = dataset.sample_weight
        if self.weights is None:
            self.positive_weights = None
            self.positives = int(count_nonzero(self.labels))
            self.total = len(self.labels)
        else:
            self.positive_weights = self.weights * self.labels
            self.positives = int(self.positive_weights.sum())
            self.total = int(self.weights.sum())

    def __call__(self, y_pred: ndarray) -> tuple[int, int, int, int]:
        if self.weights is None:
            tp = int(count_nonzero(y_pred & self.labels))
            predicted = int(count_nonzero(y_pred))
        else:
            tp = int(dot(y_pred, self.positive_weights))
            predicted = int(dot(y_pred, self.weights))
        return (
            tp,
            predicted - tp,
            self.positives - tp,
            self.total - self.positives - predicted + tp,
        )
