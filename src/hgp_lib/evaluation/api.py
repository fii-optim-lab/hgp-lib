from typing import Any

import numpy as np

from ..rules import Rule
from .backend import EvaluationBackend
from .numpy import NumpyBackend


def predict(
    rule: Rule,
    data: Any,
    *,
    backend: EvaluationBackend | None = None,
) -> np.ndarray:
    """Evaluate a rule on arbitrary data."""
    selected_backend = backend or NumpyBackend()
    predictions = selected_backend.predict(rule, data)
    return selected_backend.to_numpy(predictions)


def score(
    rule: Rule,
    data: Any,
    labels: Any,
    score_fn,
    *,
    backend: EvaluationBackend | None = None,
) -> float:
    """Evaluate and score one rule on arbitrary data."""
    raise NotImplementedError()