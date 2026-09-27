"""
Scoring functions on NumPy predictions, and the scorer used during training.

``fast_f1_score``, ``fast_accuracy_score`` and ``confusion_matrix`` work on any boolean
predictions, for use outside the library. During training, backends recognize the two
scoring functions and replace them with their own kernels, bound to the training data.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from ..utils.validation import validate_callable
from ..utils.warnings import warn_once

ScoreFn = Callable[..., float]
"""A scoring function ``score_fn(y_true, y_pred) -> float``, optionally accepting ``sample_weight``."""


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
    ``sample_weight`` (for example ``**kwargs``), the scorer is called on a tiny input.

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
    except TypeError:
        return False


@dataclass(frozen=True, slots=True)
class Scorer:
    """
    A scoring function, and whether training may merge duplicate rows for it.

    Created by `resolve_scorer`. Rows are only merged into ``sample_weight`` when
    ``merge_rows`` is ``True``, which requires ``fn`` to accept ``sample_weight``.
    Calling the scorer passes ``sample_weight`` only when it is not ``None``, so a
    scorer without weight support is never called with weights.

    Attributes:
        fn (ScoreFn): The scoring function, ``fn(y_true, y_pred) -> float``.
        merge_rows (bool): Whether duplicate rows may be merged into weights.
            Default: `False`.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Scorer, fast_f1_score
        >>> scorer = Scorer(fast_f1_score, merge_rows=True)
        >>> y_true = np.array([True, False, True])
        >>> y_pred = np.array([True, False, False])
        >>> scorer(y_true, y_pred)
        0.6666666666666666
        >>> scorer(y_true, y_pred, np.array([2, 1, 1]))
        0.8
    """

    fn: ScoreFn
    merge_rows: bool = False

    def __call__(
        self, y_true: ndarray, y_pred: ndarray, sample_weight: ndarray | None = None
    ) -> float:
        if sample_weight is None:
            return self.fn(y_true, y_pred)
        return self.fn(y_true, y_pred, sample_weight=sample_weight)


def resolve_scorer(
    score_fn: ScoreFn | None = None, optimize: bool | None = None
) -> Scorer:
    """
    Decide which scoring function to use and whether rows may be merged for it.

    - ``score_fn=None`` means `fast_f1_score`.
    - ``optimize=None`` merges rows for the built-in scorers only.
    - ``optimize=True`` merges rows if ``score_fn`` accepts ``sample_weight``. If it does
      not, a ``FutureWarning`` is emitted once and rows are not merged.
    - ``optimize=False`` never merges rows.

    Args:
        score_fn (ScoreFn | None): The user's scoring function. Default: `None`.
        optimize (bool | None): The ``optimize_scorer`` setting. Default: `None`.

    Returns:
        Scorer: The resolved scorer.

    Examples:
        >>> from hgp_lib.evaluation.scorers import resolve_scorer
        >>> resolve_scorer().merge_rows
        True
        >>> def custom(y_true, y_pred, sample_weight=None): return 0.0
        >>> resolve_scorer(custom).merge_rows
        False
        >>> resolve_scorer(custom, optimize=True).merge_rows
        True
    """
    fn = fast_f1_score if score_fn is None else score_fn
    validate_callable(fn)
    builtin = any(fn is builtin_fn for builtin_fn in BUILTIN_SCORERS)
    if optimize is None:
        return Scorer(fn, merge_rows=builtin)
    if optimize and not builtin and not accepts_sample_weight(fn):
        name = getattr(fn, "__qualname__", repr(fn))
        warn_once(
            FutureWarning(
                f"optimize_scorer=True, but {name} does not accept sample_weight, so "
                "duplicate rows are not merged. A future version will raise a "
                "ValueError instead; pass optimize_scorer=None or False to keep "
                "this behavior."
            ),
            stacklevel=2,
        )
        return Scorer(fn, merge_rows=False)
    return Scorer(fn, merge_rows=bool(optimize))
