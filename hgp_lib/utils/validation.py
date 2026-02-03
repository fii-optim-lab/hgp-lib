import inspect
from typing import Sequence, Tuple, Type, Callable, Any
import numpy as np

from ..rules import Rule


def validate_callable(maybe_callable: Callable, error_message: str | None = None):
    if not callable(maybe_callable):
        if error_message is None:
            error_message = f"score_fn must be callable, is {type(maybe_callable)}"
        raise TypeError(error_message)


def check_isinstance(value: Any, expected_type: Type | Tuple[Type, ...]):
    if not isinstance(value, expected_type):
        name = "<unknown value>"
        # Search the name in the caller
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back
            for var_name, var_val in {**frame.f_locals, **frame.f_globals}.items():
                if var_val is value:
                    name = var_name
                    break
        if isinstance(expected_type, tuple):
            expected_type = " or ".join([str(t) for t in expected_type])
        else:
            expected_type = str(expected_type)
        raise TypeError(
            f"{name} should be of type {expected_type}, but is {type(value)}"
        )


def validate_num_literals(num_literals: int):
    check_isinstance(num_literals, int)
    if num_literals <= 1:
        raise ValueError(
            f"Number of literals must be greater than 1, is '{num_literals}'"
        )


def validate_operator_types(operator_types: Sequence[Type[Rule]]):
    check_isinstance(operator_types, Sequence)
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

    check_isinstance(X, np.ndarray)
    check_isinstance(y, np.ndarray)

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length. Got X={len(X)}, y={len(y)}"
        )

    if len(X) == 0:
        raise ValueError("X and y cannot be empty")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array (samples, features), got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array (samples), got shape {y.shape}")


def validate_trainer_params(
    score_fn: Callable,
    num_epochs: int,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_every: int,
    regeneration_patience: int,
    val_score_fn: Callable | None,
):
    """
    Validate common parameters shared by GPTrainer and GPBenchmarker.

    This function consolidates validation logic for parameters that are used
    by both the trainer and benchmarker classes.

    Args:
        score_fn (Callable): Scoring function (must be callable).
        num_epochs (int): Number of training epochs (must be positive).
        train_data (np.ndarray): Training data array (2D).
        train_labels (np.ndarray): Training labels array (1D).
        val_every (int): Validation frequency (must be positive).
        regeneration_patience (int): Regeneration patience (must be positive).
        val_score_fn (Callable | None): Optional validation scoring function.

    Raises:
        TypeError: If any parameter has an incorrect type.
        ValueError: If any parameter has an invalid value.
    """
    validate_callable(score_fn)
    check_isinstance(num_epochs, int)
    if num_epochs < 1:
        raise ValueError("num_epochs must be a positive integer")

    check_X_y(train_data, train_labels)

    if val_score_fn is not None:
        validate_callable(val_score_fn)

    check_isinstance(val_every, int)
    if val_every < 1:
        raise ValueError("val_every must be a positive integer")

    check_isinstance(regeneration_patience, int)
    if regeneration_patience < 1:
        raise ValueError("regeneration_patience must be a positive integer")
