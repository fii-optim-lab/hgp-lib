import inspect
import warnings
from functools import partial
from typing import Callable, Set

import numpy as np
from numpy import ndarray

from hgp_lib.utils.validation import validate_callable


# TODO: Add documentation
# TODO: Add tests

# Track scorers that have already been warned about missing sample_weight support
_warned_scorers: Set[int] = set()


def fast_f1_score(
    y_pred: ndarray,
    y_true: ndarray,
    sample_weight: ndarray | None = None,
) -> float:
    """
    Compute F1 score with optional sample weights.

    This function supports the optimize_scorer feature of BooleanGP
    by accepting sample_weight parameter. It's optimized for boolean arrays.

    Args:
        y_pred: Boolean predictions array.
        y_true: True labels array.
        sample_weight: Optional sample weights for weighted F1.

    Returns:
        F1 score as float in [0, 1].

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.utils.metrics import fast_f1_score
        >>> y_pred = np.array([True, True, False, False])
        >>> y_true = np.array([True, False, False, True])
        >>> fast_f1_score(y_pred, y_true)
        0.5
    """
    if sample_weight is None:
        y_pred_sum = y_pred.sum()
        y_true_sum = y_true.sum()
    else:
        y_pred_sum = np.dot(y_pred, sample_weight)
        y_true_sum = np.dot(y_true, sample_weight)

    if y_pred_sum == 0 or y_true_sum == 0:
        if y_pred_sum == 0 and y_true_sum == 0:
            return 1.0
        return 0.0

    if sample_weight is None:
        return float(2 * (y_pred & y_true).sum() / (y_pred_sum + y_true_sum))
    return float(
        2 * np.dot((y_pred & y_true), sample_weight) / (y_pred_sum + y_true_sum)
    )


def accepts_sample_weight(scorer: Callable) -> bool:
    """
    Check if a scorer function accepts a sample_weight parameter.

    First checks the function signature, then falls back to a runtime test.

    Args:
        scorer (Callable): The scoring function to check.

    Returns:
        True if the scorer accepts sample_weight, False otherwise.
    """
    try:
        sig = inspect.signature(scorer)
        for param in sig.parameters.values():
            if param.name == "sample_weight":
                return True

    except (TypeError, ValueError):
        pass

    try:
        labels = np.array([1, 0, 1], dtype=bool)
        count = np.array([2, 1, 1])
        scorer(labels, labels, sample_weight=count)
        return True
    except TypeError:
        return False


def transform_duplicates_to_sample_weight(data: ndarray, labels: ndarray):
    """
    Transform data by removing duplicates and computing sample weights.

    Args:
        data (ndarray): Input data array (2D).
        labels (ndarray): Labels array (1D).

    Returns:
        Tuple of (unique_data, unique_labels, sample_weights).
    """
    Xy = np.hstack((data, labels[:, None]))
    Xy_unique, sample_weight = np.unique(Xy, axis=0, return_counts=True)
    return Xy_unique[:, :-1], Xy_unique[:, -1], sample_weight


def optimize_scorer_for_data(
    scorer: Callable[[ndarray, ndarray], float], data: ndarray, labels: ndarray
):
    """
    Optimize a scorer for the given data by deduplicating and using sample weights.

    If the scorer supports sample_weight, duplicates are removed and weights are
    computed. Otherwise, a warning is issued (once per scorer) and the original
    scorer/data are returned.

    Args:
        scorer (Callable[[ndarray, ndarray], float]): Scoring function (predictions, labels) -> float.
        data (ndarray): Input data array (2D).
        labels (ndarray): Labels array (1D).

    Returns:
        Tuple of (optimized_scorer, optimized_data, optimized_labels).
    """
    validate_callable(scorer)
    if not accepts_sample_weight(scorer):
        # Only warn once per scorer function to avoid repeated warnings
        scorer_id = id(scorer)
        if scorer_id not in _warned_scorers:
            _warned_scorers.add(scorer_id)
            warnings.warn(
                'The scorer must accept "sample_weight" to be optimized by '
                "removing duplicates in the data. Scorer optimization is disabled "
                "for this scorer.",
                stacklevel=2,
            )
    else:
        data, labels, sample_weight = transform_duplicates_to_sample_weight(
            data, labels
        )
        scorer = partial(scorer, sample_weight=sample_weight)
    return scorer, data, labels


def f1_score_multiclass(predictions, labels, sample_weight=None):
    """
    Compute F1 score for multiclass classification with support for missing predictions (-1).

    Args:
        predictions (np.ndarray): Predicted class labels (may contain -1 for missing).
        labels (np.ndarray): True class labels.
        sample_weight (np.ndarray, optional): Not used, for API compatibility.

    Returns:
        float: F1 score in [0, 1].

    Notes:
        - If both predictions and labels are empty, returns 1.0 (perfect agreement on empty set).
        - If all predictions are missing but labels exist, returns 0.0.
    """
    if len(predictions) == 0 and len(labels) == 0:
        return 1.0
    mask = (predictions != -1) & (labels != -1)
    y_pred = predictions[mask]
    y_true = labels[mask]
    tp = np.sum(y_pred == y_true)
    pred_sum = len(y_pred)
    label_sum = len(y_true)
    if pred_sum == 0 or label_sum == 0:
        return 0.0
    precision = tp / pred_sum
    recall = tp / label_sum
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
