import numpy as np

from ...rules import Rule


def predict_rule(
    rule: Rule,
    data: np.ndarray,
) -> np.ndarray:
    """Evaluate one rule using NumPy."""
    raise NotImplementedError