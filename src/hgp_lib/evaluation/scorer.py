import inspect
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy import ndarray

from ..utils.validation import validate_callable

# Track scorers that have already been warned about missing sample_weight support
_warned_scorers: set[int] = set()


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
) -> tuple[int, int, int, int]:
    """
    Compute confusion matrix values from boolean label and prediction arrays.

    Args:
        y_true (np.ndarray):
            Boolean ground-truth labels.
        y_pred (np.ndarray):
            Boolean predictions.
        sample_weight (np.ndarray | None):
            Optional per-sample weights. Default: `None`.

    Returns:
        tuple[int, int, int, int]: ``(tp, fp, fn, tn)``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.scorer import confusion_matrix
        >>> y_true = np.array([True, False, True, False])
        >>> y_pred = np.array([True, True, False, False])
        >>> confusion_matrix(y_true, y_pred)
        (1, 1, 1, 1)
    """
    if sample_weight is None:
        tp = np.count_nonzero(y_pred & y_true)
        fp = np.count_nonzero(y_pred & ~y_true)
        total_true = np.count_nonzero(y_true)
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
    y_true: ndarray,
    y_pred: ndarray,
    sample_weight: ndarray | None = None,
) -> float:
    """
    Compute F1 score with optional sample weights.

    This function supports the optimize_scorer feature of BooleanGP
    by accepting sample_weight parameter. It's optimized for boolean arrays.

    Args:
        y_true: True labels array.
        y_pred: Boolean predictions array.
        sample_weight: Optional sample weights for weighted F1.

    Returns:
        F1 score as float in [0, 1].

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.scorer import fast_f1_score
        >>> y_pred = np.array([True, True, False, False])
        >>> y_true = np.array([True, False, False, True])
        >>> fast_f1_score(y_true, y_pred)
        0.5
    """
    if sample_weight is None:
        y_pred_sum = np.count_nonzero(y_pred)
        y_true_sum = np.count_nonzero(y_true)
        if y_pred_sum == 0 or y_true_sum == 0:
            return 1.0 if y_pred_sum == 0 and y_true_sum == 0 else 0.0
        return float(2 * np.count_nonzero(y_pred & y_true) / (y_pred_sum + y_true_sum))

    y_pred_sum = np.dot(y_pred, sample_weight)
    y_true_sum = np.dot(y_true, sample_weight)
    if y_pred_sum == 0 or y_true_sum == 0:
        return 1.0 if y_pred_sum == 0 and y_true_sum == 0 else 0.0
    return float(2 * np.dot(y_pred & y_true, sample_weight) / (y_pred_sum + y_true_sum))


def fast_accuracy_score(
    y_true: ndarray,
    y_pred: ndarray,
    sample_weight: ndarray | None = None,
) -> float:
    """
    Compute accuracy with optional sample weights.

    This function supports the optimize_scorer feature of BooleanGP
    by accepting sample_weight parameter. It's optimized for boolean arrays.

    Args:
        y_true: True labels array.
        y_pred: Boolean predictions array.
        sample_weight: Optional sample weights for weighted F1.

    Returns:
        Accuracy as float in [0, 1].

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.scorer import fast_accuracy_score
        >>> y_pred = np.array([True, True, False, False])
        >>> y_true = np.array([True, False, False, True])
        >>> fast_accuracy_score(y_true, y_pred)
        0.5
    """
    correct = y_true == y_pred
    if sample_weight is None:
        return float(correct.mean())
    return float(np.dot(correct, sample_weight) / sample_weight.sum())


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
        >>> from hgp_lib.evaluation.scorer import accepts_sample_weight
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


def transform_duplicates_to_sample_weight(
    data: ndarray, labels: ndarray, sample_weight: ndarray | None = None
):
    """
    Remove duplicate rows from ``(data, labels)`` and return sample weights.

    Rows that appear multiple times are collapsed into a single row with a
    weight equal to the original count, or to the sum of their weights when
    ``sample_weight`` is given.

    Args:
        data (ndarray):
            2-D input data.
        labels (ndarray):
            1-D label array (same length as ``data``).
        sample_weight (ndarray | None):
            Optional weights of the input rows, e.g. rows that were already
            deduplicated. Default: `None`.

    Returns:
        tuple[ndarray, ndarray, ndarray | None]: ``(unique_data, unique_labels, sample_weights)``.
            When there are no duplicates, the inputs are returned unchanged, with
            ``sample_weight`` as the weights.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.scorer import transform_duplicates_to_sample_weight
        >>> data = np.array([[1, 0], [1, 0], [0, 1]])
        >>> labels = np.array([1, 1, 0])
        >>> ud, ul, sw = transform_duplicates_to_sample_weight(data, labels)
        >>> len(ud) < len(data)
        True
        >>> bool(sw.sum() == len(data))
        True
        >>> _, _, sw = transform_duplicates_to_sample_weight(data, labels, np.array([2, 3, 1]))
        >>> sw.tolist()
        [1, 5]
    """
    Xy_packed = np.ascontiguousarray(
        np.packbits(np.hstack((data, labels[:, None])), axis=1)
    )

    row_dtype = np.dtype((np.void, Xy_packed.shape[1]))
    row_view = Xy_packed.view(row_dtype).ravel()

    if sample_weight is None:
        _, unique_idx, counts = np.unique(
            row_view,
            return_index=True,
            return_counts=True,
        )
    else:
        _, unique_idx, inverse = np.unique(
            row_view,
            return_index=True,
            return_inverse=True,
        )
        counts = np.bincount(inverse, weights=sample_weight).astype(sample_weight.dtype)
    if len(unique_idx) == len(labels):
        return data, labels, sample_weight

    return data[unique_idx], labels[unique_idx], counts


def select_weighted_rows(sample_weight: ndarray, indices: ndarray):
    """
    Map indices of original rows to deduplicated rows and their new weights.

    Deduplicated row ``i`` stands for ``sample_weight[i]`` consecutive original rows,
    so drawing indices from ``range(sample_weight.sum())`` samples original rows
    without expanding the data.

    Args:
        sample_weight (ndarray):
            Integer weights of the deduplicated rows.
        indices (ndarray):
            Distinct indices of original rows, in ``[0, sample_weight.sum())``.

    Returns:
        tuple[ndarray, ndarray]: ``(rows, weights)``, the deduplicated rows that were
            selected and how many selected original rows each of them stands for.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.utils.metrics import select_weighted_rows
        >>> rows, weights = select_weighted_rows(np.array([3, 1, 2]), np.array([0, 2, 4, 5]))
        >>> rows.tolist(), weights.tolist()
        ([0, 2], [2, 2])
    """
    owners = np.searchsorted(np.cumsum(sample_weight), indices, side="right")
    counts = np.bincount(owners, minlength=len(sample_weight))
    rows = np.flatnonzero(counts)
    return rows, counts[rows]


class SampleWeightScorer:
    """
    Adapter that binds a fixed ``sample_weight`` to a scorer.

    Wraps a scorer of the form ``scorer(y_true, y_pred, sample_weight=...)`` and exposes
    a two-argument callable ``scorer(y_true, y_pred)`` that injects the stored weights.
    It is used by :func:`optimize_scorers_for_data` after duplicate rows are collapsed
    into per-row weights, so the wrapped scorer returns the same value it would on the
    original, un-deduplicated data.

    Args:
        scorer (Callable):
            A scorer accepting ``(y_true, y_pred, sample_weight=...)`` and returning a
            float, following the library's ``(y_true, y_pred)`` argument order.
        sample_weight (ndarray):
            Per-row weights bound to every call.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.scorer import SampleWeightScorer, fast_f1_score
        >>> weighted = SampleWeightScorer(fast_f1_score, np.array([2, 1, 1]))
        >>> y_true = np.array([True, False, True])
        >>> y_pred = np.array([True, False, False])
        >>> round(weighted(y_true, y_pred), 3)
        0.8
    """

    def __init__(
        self, scorer: Callable[[ndarray, ndarray], Any], sample_weight: ndarray
    ):
        self.scorer = scorer
        self.sample_weight = sample_weight

    def __call__(self, y_true: ndarray, y_pred: ndarray):
        return self.scorer(y_true, y_pred, sample_weight=self.sample_weight)




def optimize_scorers_for_data(
    *scorers: Callable[[ndarray, ndarray], Any],
    data: ndarray,
    labels: ndarray,
    sample_weight: ndarray | None = None,
):
    """
    Optimise scorers by deduplicating data and binding ``sample_weight``.

    If every scorer accepts ``sample_weight``, duplicate rows are removed and
    each scorer is wrapped with ``SampleWeightScorer`` to inject the computed
    weights. Otherwise, a warning is issued (once per scorer) and the original
    data is returned unchanged.

    Args:
        *scorers (Callable[[ndarray, ndarray], Any]):
            One or more scoring functions.
        data (ndarray):
            2-D input data.
        labels (ndarray):
            1-D label array.
        sample_weight (ndarray | None):
            Optional weights of the input rows, e.g. rows that were already
            deduplicated. Duplicates add up their weights. Default: `None`.

    Returns:
        tuple: ``(*optimised_scorers, data, labels)``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.scorer import optimize_scorers_for_data
        >>> from sklearn.metrics import accuracy_score
        >>> data = np.array([[1, 0], [1, 0], [0, 1]])
        >>> labels = np.array([1, 1, 0])
        >>> opt_acc, opt_data, opt_labels = optimize_scorers_for_data(accuracy_score, data=data, labels=labels)
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
            data, labels, sample_weight
        )
        if sample_weight is not None:
            scorers = [SampleWeightScorer(scorer, sample_weight) for scorer in scorers]
    return *scorers, data, labels
