import inspect
import warnings
from functools import partial
from typing import Callable, Set, Tuple, Any

import numpy as np
from numpy import ndarray

from hgp_lib.utils.validation import validate_callable


# Track scorers that have already been warned about missing sample_weight support
_warned_scorers: Set[int] = set()


def confusion_matrix(
    y_pred: np.ndarray, y_true: np.ndarray, sample_weight: np.ndarray | None = None
) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix values from boolean prediction and label arrays.

    Args:
        y_pred (np.ndarray):
            Boolean predictions.
        y_true (np.ndarray):
            Boolean ground-truth labels.
        sample_weight (np.ndarray | None):
            Optional per-sample weights. Default: `None`.

    Returns:
        Tuple[int, int, int, int]: ``(tp, fp, fn, tn)``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.utils.metrics import confusion_matrix
        >>> y_pred = np.array([True, True, False, False])
        >>> y_true = np.array([True, False, True, False])
        >>> confusion_matrix(y_pred, y_true)
        (1, 1, 1, 1)
    """
    if sample_weight is None:
        tp = (y_pred & y_true).sum()
        fp = (y_pred & ~y_true).sum()
        total_true = y_true.sum()
        fn = total_true - tp
        tn = len(y_pred) - total_true - fp
    else:
        tp = ((y_pred & y_true) * sample_weight).sum()
        fp = ((y_pred & ~y_true) * sample_weight).sum()
        total_true = (y_true * sample_weight).sum()
        fn = total_true - tp
        tn = sample_weight.sum() - total_true - fp
    return int(tp), int(fp), int(fn), int(tn)


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
    Check if a scorer function accepts a ``sample_weight`` parameter.

    Inspects the function signature first; falls back to a runtime probe if
    signature inspection fails.

    Args:
        scorer (Callable):
            The scoring function to check.

    Returns:
        bool: ``True`` if the scorer accepts ``sample_weight``.

    Examples:
        >>> from hgp_lib.utils.metrics import accepts_sample_weight
        >>> def with_sw(p, l, sample_weight=None): return 0.0
        >>> accepts_sample_weight(with_sw)
        True
        >>> def without_sw(p, l): return 0.0
        >>> accepts_sample_weight(without_sw)
        False
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
    Remove duplicate rows from ``(data, labels)`` and return sample weights.

    Rows that appear multiple times are collapsed into a single row with a
    weight equal to the original count.

    Args:
        data (ndarray):
            2-D input data.
        labels (ndarray):
            1-D label array (same length as ``data``).

    Returns:
        Tuple[ndarray, ndarray, ndarray]: ``(unique_data, unique_labels, sample_weights)``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.utils.metrics import transform_duplicates_to_sample_weight
        >>> data = np.array([[1, 0], [1, 0], [0, 1]])
        >>> labels = np.array([1, 1, 0])
        >>> ud, ul, sw = transform_duplicates_to_sample_weight(data, labels)
        >>> len(ud) < len(data)
        True
        >>> bool(sw.sum() == len(data))
        True
    """
    Xy = np.hstack((data, labels[:, None]))
    Xy_unique, sample_weight = np.unique(Xy, axis=0, return_counts=True)
    return Xy_unique[:, :-1], Xy_unique[:, -1], sample_weight


def optimize_scorers_for_data(
    *scorers: Callable[[ndarray, ndarray], Any], data: ndarray, labels: ndarray
):
    """
    Optimise scorers by deduplicating data and binding ``sample_weight``.

    If every scorer accepts ``sample_weight``, duplicate rows are removed and
    each scorer is wrapped with ``functools.partial`` to inject the computed
    weights. Otherwise a warning is issued (once per scorer) and the original
    data is returned unchanged.

    Args:
        *scorers (Callable[[ndarray, ndarray], Any]):
            One or more scoring functions.
        data (ndarray):
            2-D input data.
        labels (ndarray):
            1-D label array.

    Returns:
        Tuple: ``(*optimised_scorers, data, labels)``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.utils.metrics import optimize_scorers_for_data
        >>> def acc(p, l, sample_weight=None): return float((p == l).mean())
        >>> data = np.array([[1, 0], [1, 0], [0, 1]])
        >>> labels = np.array([1, 1, 0])
        >>> opt_acc, opt_data, opt_labels = optimize_scorers_for_data(acc, data=data, labels=labels)
        >>> len(opt_data) <= len(data)
        True
    """
    scorers_ok = True
    for scorer in scorers:
        validate_callable(scorer)
        if not accepts_sample_weight(scorer):
            scorers_ok = False
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
    if scorers_ok:
        data, labels, sample_weight = transform_duplicates_to_sample_weight(
            data, labels
        )
        scorers = [partial(scorer, sample_weight=sample_weight) for scorer in scorers]
    return *scorers, data, labels
