import numpy as np
from numpy import ndarray

from ..rules import Rule
from .backend import EvaluationBackend
from .numpy import NumpyBackend
from .scorers import ScoreFn, fast_f1_score


def _check_binarized(data: ndarray) -> None:
    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"data must be a binarized boolean NumPy array, is {type(data).__name__}. "
            "For raw data, use BooleanRuleClassifier or GPBenchmarker, which binarize "
            "it with their fitted binarizer."
        )
    if data.dtype != np.bool_:
        raise TypeError(
            f"data must be a binarized boolean array, has dtype {data.dtype}. "
            "For raw data, use BooleanRuleClassifier or GPBenchmarker, which binarize "
            "it with their fitted binarizer."
        )
    if data.ndim != 2:
        raise ValueError(f"data must be a 2-D array, has shape {data.shape}")


def predict(
    rule: Rule, data: ndarray, *, backend: EvaluationBackend | None = None
) -> ndarray:
    """
    Evaluate a rule on binarized data.

    ``data`` must be the binarized boolean matrix, with the same feature columns the rule
    was trained on (for example the output of a fitted binarizer). For raw data, use
    `BooleanRuleClassifier.predict` or `GPBenchmarker.predict` instead. The data is
    neither copied nor merged.

    Args:
        rule (Rule): The rule to evaluate.
        data (ndarray): 2-D boolean array, rows are instances and columns are features.
        backend (EvaluationBackend | None): The backend to evaluate with. ``None`` uses
            ``NumpyBackend()``. Default: `None`.

    Returns:
        ndarray: 1-D boolean array, one prediction per row.

    Raises:
        TypeError: If ``data`` is not a boolean NumPy array.
        ValueError: If ``data`` is not 2-D.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import predict
        >>> from hgp_lib.rules import And, Literal
        >>> data = np.array([[True, False], [True, True], [False, False]])
        >>> predict(And([Literal(value=0), Literal(value=1, negated=True)]), data)
        array([ True, False, False])
    """
    _check_binarized(data)
    if backend is None:
        backend = NumpyBackend()
    return backend.predict(rule, data)


def score(
    rule: Rule,
    data: ndarray,
    labels: ndarray,
    score_fn: ScoreFn | None = None,
    *,
    backend: EvaluationBackend | None = None,
) -> float:
    """
    Score a rule on binarized data.

    Same data requirements as `predict`. The rows are scored as given, without merging.

    Args:
        rule (Rule): The rule to score.
        data (ndarray): 2-D boolean array, rows are instances and columns are features.
        labels (ndarray): 1-D binary labels, one per row.
        score_fn (ScoreFn | None): Scoring function ``score_fn(y_true, y_pred)``.
            ``None`` uses `fast_f1_score`. Default: `None`.
        backend (EvaluationBackend | None): The backend to evaluate with. ``None`` uses
            ``NumpyBackend()``. Default: `None`.

    Returns:
        float: The score.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import fast_accuracy_score, score
        >>> from hgp_lib.rules import Literal
        >>> data = np.array([[True], [True], [False], [False]])
        >>> labels = np.array([1, 0, 0, 0])
        >>> score(Literal(value=0), data, labels)
        0.6666666666666666
        >>> score(Literal(value=0), data, labels, fast_accuracy_score)
        0.75
    """
    fn = fast_f1_score if score_fn is None else score_fn
    return float(fn(labels, predict(rule, data, backend=backend)))
