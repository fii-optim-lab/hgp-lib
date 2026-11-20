from numpy import ndarray

from ..rules import Rule
from ..utils.validation import check_X, check_X_y
from .backend import EvaluationBackend
from .dataset import Dataset
from .numpy import NumpyBackend
from .scorers import ScoreFn, fast_f1_score


def predict(
    rule: Rule, data: ndarray, *, backend: EvaluationBackend | None = None
) -> ndarray:
    """
    Evaluate a rule on binarized data.

    ``data`` must be the binarized boolean matrix, with the same feature columns the rule
    was trained on (for example the output of a fitted binarizer). For raw data, use
    `BooleanRuleClassifier.predict` or `GPBenchmarker.predict` instead. The rows are not
    merged.

    Args:
        rule (Rule): The rule to evaluate.
        data (ndarray): 2-D boolean array, rows are instances and columns are features.
        backend (EvaluationBackend | None): The backend to evaluate with. ``None`` uses
            ``NumpyBackend()``. Default: `None`.

    Returns:
        ndarray: 1-D boolean NumPy array, one prediction per row.

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
    check_X(data)
    if backend is None:
        backend = NumpyBackend()
    return backend.to_numpy(backend.predict(rule, backend.from_numpy(data)))


def score(
    rule: Rule,
    data: ndarray,
    labels: ndarray,
    score_fn: ScoreFn = fast_f1_score,
    *,
    backend: EvaluationBackend | None = None,
) -> float:
    """
    Score a rule on binarized data.

    Same data requirements as `predict`. The rows are scored as given, without merging,
    by the backend, so the score has the backend's ``dtype`` precision. With
    `TorchBackend`, a custom ``score_fn`` receives tensors.

    Args:
        rule (Rule): The rule to score.
        data (ndarray): 2-D boolean array, rows are instances and columns are features.
        labels (ndarray): 1-D binary labels, one per row: booleans, or integers 0 and 1.
        score_fn (ScoreFn): Scoring function ``score_fn(y_true, y_pred)``.
            Default: `fast_f1_score`.
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
        0.6666666865348816
        >>> score(Literal(value=0), data, labels, fast_accuracy_score)
        0.75
    """
    labels = check_X_y(data, labels)
    if backend is None:
        backend = NumpyBackend()
    evaluator = backend.bind(Dataset(data, labels), score_fn)
    return float(backend.to_numpy(evaluator.score([rule]))[0])
