from typing import Sequence, Type
import numpy as np

from ..rules import Rule


def validate_num_literals(num_literals: int):
    if not isinstance(num_literals, int):
        raise TypeError(
            f"Number of literals must be an integer, is '{type(num_literals)}'"
        )
    if num_literals <= 1:
        raise ValueError(
            f"Number of literals must be greater than 1, is '{num_literals}'"
        )


def validate_operator_types(operator_types: Sequence[Type[Rule]]):
    if not isinstance(operator_types, Sequence):
        raise TypeError(
            f"operator_types must be a Sequence, is '{type(operator_types)}'"
        )
    if len(operator_types) < 2:
        raise ValueError("operator_types must have at least two operator types")
    for operator_type in operator_types:
        if not issubclass(operator_type, Rule):
            raise TypeError(
                f"All operator types must be subclassing Rule. Found '{type(operator_type)}'"
            )


def check_X_y(X: np.ndarray, y: np.ndarray):
    """
    Validate input data and labels.

    Checks that X and y are numpy arrays, have compatible shapes (same number of samples),
    and are not None.

    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): Target labels.

    Raises:
        ValueError: If X or y is None, empty, or have mismatched lengths.
        TypeError: If X or y is not a numpy array.
    """
    if X is None:
        raise ValueError("X (data) cannot be None")
    if y is None:
        raise ValueError("y (labels) cannot be None")

    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be a numpy array, got {type(X)}")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be a numpy array, got {type(y)}")

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length. Got X={len(X)}, y={len(y)}"
        )

    if len(X) == 0:
        raise ValueError("X and y cannot be empty")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array (samples, features), got shape {X.shape}")
