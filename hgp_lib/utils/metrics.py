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


def normalize(scores: ndarray) -> ndarray:
    """
    Normalize scores to [0, 1] range.

    Args:
        scores (ndarray): 1D array of scores.

    Returns:
        ndarray: Scores scaled so that min maps to 0 and max to 1.
            If all values are equal, returns array of ones.
            Returns a copy; does not modify the input.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.utils.metrics import normalize
        >>> normalize(np.array([1.0, 2.0, 3.0]))
        array([0. , 0.5, 1. ])
        >>> normalize(np.array([0.5, 0.5]))
        array([1., 1.])
    """
    if len(scores) == 0:
        return scores.copy()
    min_s = float(np.min(scores))
    max_s = float(np.max(scores))
    if max_s == min_s:
        return np.ones_like(scores, dtype=float)
    return (scores.astype(float) - min_s) / (max_s - min_s)
