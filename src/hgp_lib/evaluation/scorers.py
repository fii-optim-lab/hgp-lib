"""
Scoring functions on NumPy predictions, and the rule that decides when rows are merged.

``fast_f1_score``, ``fast_accuracy_score`` and ``confusion_matrix`` work on any boolean
predictions, for use outside the library. During training, backends recognize the two
scoring functions and replace them with their own kernels, bound to the training data.

A scoring function returns a finite float for every prediction, never NaN. Selection
and best-rule tracking compare scores, and comparisons with NaN are always false.
"""

import inspect
from collections.abc import Callable

import numpy as np
from numpy import ndarray

ScoreFn = Callable[..., float]
"""
A scoring function ``score_fn(y_true, y_pred) -> float``, optionally accepting
``sample_weight``, and optionally with a ``batched(y_true, y_preds) -> scores`` method
that scores a 2-D stack of predictions, one row per rule.
"""


def fast_f1_score(
    y_true: ndarray,
    y_pred: ndarray,
    sample_weight: ndarray | None = None,
) -> float:
    """
    Compute the F1 score of boolean predictions, with optional sample weights.

    When there are no positive labels and no positive predictions, the score is ``1.0``.

    Args:
        y_true (ndarray): Binary labels (boolean or 0/1 integers).
        y_pred (ndarray): Boolean predictions.
        sample_weight (ndarray | None): Optional per-row weights. Default: `None`.

    Returns:
        float: F1 score in ``[0, 1]``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import fast_f1_score
        >>> y_true = np.array([True, False, False, True])
        >>> y_pred = np.array([True, True, False, False])
        >>> fast_f1_score(y_true, y_pred)
        0.5
        >>> fast_f1_score(y_true, y_pred, sample_weight=np.array([3, 1, 1, 1]))
        0.75
    """
    if sample_weight is None:
        y_pred_sum = np.count_nonzero(y_pred)
        y_true_sum = np.count_nonzero(y_true)
        if y_true_sum == 0:
            return 1.0 if y_pred_sum == 0 else 0.0
        return float(2 * np.count_nonzero(y_pred & y_true) / (y_pred_sum + y_true_sum))

    y_pred_sum = np.dot(y_pred, sample_weight)
    y_true_sum = np.dot(y_true, sample_weight)
    if y_true_sum == 0:
        return 1.0 if y_pred_sum == 0 else 0.0
    return float(2 * np.dot(y_pred & y_true, sample_weight) / (y_pred_sum + y_true_sum))


def fast_accuracy_score(
    y_true: ndarray,
    y_pred: ndarray,
    sample_weight: ndarray | None = None,
) -> float:
    """
    Compute the accuracy of boolean predictions, with optional sample weights.

    Args:
        y_true (ndarray): Binary labels (boolean or 0/1 integers).
        y_pred (ndarray): Boolean predictions.
        sample_weight (ndarray | None): Optional per-row weights. Default: `None`.

    Returns:
        float: Accuracy in ``[0, 1]``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import fast_accuracy_score
        >>> y_true = np.array([True, False, False, True])
        >>> y_pred = np.array([True, True, False, False])
        >>> fast_accuracy_score(y_true, y_pred)
        0.5
        >>> fast_accuracy_score(y_true, y_pred, sample_weight=np.array([3, 1, 1, 1]))
        0.6666666666666666
    """
    correct = y_true == y_pred
    if sample_weight is None:
        return float(np.count_nonzero(correct) / len(correct))
    return float(np.dot(correct, sample_weight) / sample_weight.sum())


def confusion_matrix(
    y_true: ndarray,
    y_pred: ndarray,
    sample_weight: ndarray | None = None,
) -> tuple[int, int, int, int]:
    """
    Count true positives, false positives, false negatives and true negatives.

    Args:
        y_true (ndarray): Binary labels (boolean or 0/1 integers).
        y_pred (ndarray): Boolean predictions.
        sample_weight (ndarray | None): Optional integer row counts. Default: `None`.

    Returns:
        tuple[int, int, int, int]: ``(tp, fp, fn, tn)``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import confusion_matrix
        >>> y_true = np.array([True, False, True, False])
        >>> y_pred = np.array([True, True, False, False])
        >>> confusion_matrix(y_true, y_pred)
        (1, 1, 1, 1)
        >>> confusion_matrix(y_true, y_pred, sample_weight=np.array([2, 3, 1, 1]))
        (2, 3, 1, 1)
    """
    if sample_weight is None:
        tp = int(np.count_nonzero(y_pred & y_true))
        predicted = int(np.count_nonzero(y_pred))
        positives = int(np.count_nonzero(y_true))
        total = len(y_pred)
    else:
        tp = int(np.dot(y_pred & y_true, sample_weight))
        predicted = int(np.dot(y_pred, sample_weight))
        positives = int(np.dot(y_true, sample_weight))
        total = int(sample_weight.sum())
    return tp, predicted - tp, positives - tp, total - positives - predicted + tp


BUILTIN_SCORERS = (fast_f1_score, fast_accuracy_score)
"""Scoring functions that backends replace with their own kernels."""


def accepts_sample_weight(scorer: Callable) -> bool:
    """
    Check whether a scoring function accepts a ``sample_weight`` argument.

    Inspects the signature first. When the signature cannot be read or does not name
    ``sample_weight`` (for example ``**kwargs``), the scorer is called on a tiny NumPy
    input; a scorer that fails on it is reported as not accepting ``sample_weight``.

    Args:
        scorer (Callable): The scoring function to check.

    Returns:
        bool: ``True`` if the scorer accepts ``sample_weight``.

    Examples:
        >>> from hgp_lib.evaluation import accepts_sample_weight
        >>> def with_sw(y_true, y_pred, sample_weight=None): return 0.0
        >>> accepts_sample_weight(with_sw)
        True
        >>> def without_sw(y_true, y_pred): return 0.0
        >>> accepts_sample_weight(without_sw)
        False
    """
    try:
        if "sample_weight" in inspect.signature(scorer).parameters:
            return True
    except (TypeError, ValueError):
        pass

    try:
        labels = np.array([True, False, True])
        scorer(labels, labels, sample_weight=np.array([2, 1, 1]))
        return True
    except Exception:  # noqa: BLE001 - any failure means "not supported"
        return False


def merges_rows(score_fn: ScoreFn, optimize_scorer: bool | None) -> bool:
    """
    Decide from ``optimize_scorer`` whether duplicate rows are merged for ``score_fn``.

    - ``None`` merges rows for the built-in scorers (`fast_f1_score`,
      `fast_accuracy_score`) only.
    - ``True`` merges rows for any scorer. Binding then raises a ``ValueError`` if the
      scorer does not accept ``sample_weight``.
    - ``False`` never merges rows.

    Args:
        score_fn (ScoreFn): The scoring function.
        optimize_scorer (bool | None): The ``optimize_scorer`` setting.

    Returns:
        bool: Whether rows are merged into sample weights.

    Examples:
        >>> from hgp_lib.evaluation import fast_f1_score
        >>> from hgp_lib.evaluation.scorers import merges_rows
        >>> merges_rows(fast_f1_score, None)
        True
        >>> def custom(y_true, y_pred, sample_weight=None): return 0.0
        >>> merges_rows(custom, None), merges_rows(custom, True)
        (False, True)
    """
    if optimize_scorer is None:
        return any(score_fn is builtin for builtin in BUILTIN_SCORERS)
    return optimize_scorer


def check_merge_rows(score_fn: ScoreFn) -> None:
    """
    Raise a ``ValueError`` if rows cannot be merged for ``score_fn``.

    Merged rows are scored with ``sample_weight``, so the scorer (and its ``batched``
    method, when it has one) must accept it.

    Args:
        score_fn (ScoreFn): The scoring function.

    Raises:
        ValueError: If ``score_fn`` or ``score_fn.batched`` does not accept
            ``sample_weight``.
    """
    if any(score_fn is builtin for builtin in BUILTIN_SCORERS):
        return
    batched = getattr(score_fn, "batched", None)
    for fn in (score_fn,) if batched is None else (score_fn, batched):
        if not accepts_sample_weight(fn):
            name = getattr(fn, "__qualname__", repr(fn))
            raise ValueError(
                f"Duplicate rows cannot be merged for {name}: merged rows are scored "
                "with sample_weight, and it does not accept sample_weight. Pass "
                "merge_rows=False (optimize_scorer=None or False in BooleanGPConfig), "
                "or a scorer that accepts sample_weight."
            )
