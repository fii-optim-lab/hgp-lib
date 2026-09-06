from collections.abc import Callable

from numpy import dot, ndarray

from ..scorer import transform_duplicates_to_sample_weight


def _y_true_zero(
    _y_true: ndarray,
    y_pred: ndarray,
) -> float:
    return 1.0 if y_pred.sum() == 0 else 0.0


def _f1_score(y_true: ndarray, y_pred: ndarray, y_true_sum: ndarray) -> float:
    y_pred_sum = y_pred.sum()
    if y_pred_sum == 0:
        return 0.0
    return 2 * (y_pred & y_true).sum() / (y_pred_sum + y_true_sum)


def _y_true_zero_weighted(
    _y_true: ndarray, y_pred: ndarray, sample_weight: ndarray
) -> float:
    return 1.0 if dot(y_pred, sample_weight) == 0 else 0.0


def _f1_score_weighted(
    y_true: ndarray,
    y_pred: ndarray,
    y_true_sum: ndarray,
    sample_weight: ndarray,
) -> float:
    y_pred_sum = dot(y_pred, sample_weight)
    if y_pred_sum == 0:
        return 0.0
    return 2 * dot(y_pred & y_true, sample_weight) / (y_pred_sum + y_true_sum)


class _F1Scorer:
    def __init__(self, y_true_sum: ndarray):
        self.y_true_sum = y_true_sum

    def __call__(self, y_true: ndarray, y_pred: ndarray) -> float:
        return _f1_score(y_true, y_pred, self.y_true_sum)


class _F1ZeroWeighted:
    def __init__(self, sample_weight: ndarray):
        self.sample_weight = sample_weight

    def __call__(self, y_true: ndarray, y_pred: ndarray) -> float:
        return _y_true_zero_weighted(y_true, y_pred, self.sample_weight)


class _F1WeightedScorer:
    def __init__(self, y_true_sum: ndarray, sample_weight: ndarray):
        self.y_true_sum = y_true_sum
        self.sample_weight = sample_weight

    def __call__(self, y_true: ndarray, y_pred: ndarray) -> float:
        return _f1_score_weighted(y_true, y_pred, self.y_true_sum, self.sample_weight)


def create_fast_np_f1_score(
    data: ndarray,
    labels: ndarray,
) -> Callable[[ndarray, ndarray], float]:
    data, labels, sample_weight = transform_duplicates_to_sample_weight(data, labels)
    if sample_weight is None:
        y_true_sum = labels.sum()
        if y_true_sum == 0:
            return _y_true_zero
        return _F1Scorer(y_true_sum).__call__
    y_true_sum = dot(labels, sample_weight)
    if y_true_sum == 0:
        return _F1ZeroWeighted(sample_weight).__call__
    return _F1WeightedScorer(y_true_sum, sample_weight).__call__

# TODO: Also implement batched fast f1 score