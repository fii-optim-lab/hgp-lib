from typing import Any


def confusion_matrix(
    y_true: Any, y_pred: Any, sample_weight: Any | None = None
) -> tuple[int, int, int, int]:
    """
    Compute confusion matrix values from boolean label and prediction arrays.

    Args:
        y_true (np.ndarray | torch.Tensor):
            Boolean ground-truth labels.
        y_pred (np.ndarray | torch.Tensor):
            Boolean predictions.
        sample_weight (np.ndarray | torch.Tensor | None):
            Optional per-sample weights. Default: `None`.

    Returns:
        tuple[int, int, int, int]: ``(tp, fp, fn, tn)``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.functional import confusion_matrix
        >>> y_true = np.array([True, False, True, False])
        >>> y_pred = np.array([True, True, False, False])
        >>> confusion_matrix(y_true, y_pred)
        (1, 1, 1, 1)
    """
    if sample_weight is None:
        tp = int((y_pred & y_true).sum())
        fp = int((y_pred & ~y_true).sum())
        total_true = int(y_true.sum())
        fn = total_true - tp
        tn = len(y_pred) - total_true - fp
    else:
        tp = int(((y_pred & y_true) * sample_weight).sum())
        fp = int(((y_pred & ~y_true) * sample_weight).sum())
        total_true = int((y_true * sample_weight).sum())
        fn = total_true - tp
        tn = int(sample_weight.sum()) - total_true - fp
    return tp, fp, fn, tn
