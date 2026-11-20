import inspect
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..rules import Rule
from .warnings import warn_moved


def __getattr__(name: str):
    # ComplexityCheck moved to hgp_lib.rules in 2.0.0.
    if name == "ComplexityCheck":
        from ..rules import ComplexityCheck

        warn_moved(
            "hgp_lib.utils.validation.ComplexityCheck", "hgp_lib.rules.ComplexityCheck"
        )
        return ComplexityCheck
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def validate_callable(maybe_callable: Callable, error_message: str | None = None):
    """
    Validate that a value is callable.

    Args:
        maybe_callable (Callable): Value to check.
        error_message (str | None): Optional custom error message. Default: `None`.

    Raises:
        TypeError: If value is not callable.

    Examples:
        >>> from hgp_lib.utils.validation import validate_callable
        >>> validate_callable(len)  # no error
        >>> validate_callable(42)
        Traceback (most recent call last):
        ...
        TypeError: score_fn must be callable, is <class 'int'>
    """
    if not callable(maybe_callable):
        if error_message is None:
            error_message = f"score_fn must be callable, is {type(maybe_callable)}"
        raise TypeError(error_message)


BOOL_TYPES = (bool, np.bool_)
"""Types accepted for boolean options: Python ``bool`` and NumPy ``np.bool_`` scalars."""


def check_isinstance(
    value: Any, expected_type: type | tuple[type, ...], name: str | None = None
):
    """
    Check that a value is an instance of expected type(s).

    Args:
        value (Any): Value to check.
        expected_type (type | tuple[type, ...]): Expected type or tuple of types.
        name (str | None): Name of the value in the error message. When ``None``, the
            name is looked up in the caller's variables, which does not work for
            attributes such as ``self.batched``. Default: `None`.

    Raises:
        TypeError: If value is not an instance of expected type.

    Examples:
        >>> from hgp_lib.utils.validation import check_isinstance
        >>> check_isinstance(1.5, (int, float))  # no error
        >>> check_isinstance("yes", bool, "batched")
        Traceback (most recent call last):
        ...
        TypeError: batched should be of type <class 'bool'>, but is <class 'str'>
    """
    if not isinstance(value, expected_type):
        if name is None:
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


def validate_batching(batched: bool, batch_size: int | None):
    """
    Validate the ``batched`` and ``batch_size`` options of an evaluation backend.

    Args:
        batched (bool): Whether rules are scored in batches.
        batch_size (int | None): Rules per batch, or ``None`` for one batch.

    Raises:
        TypeError: If ``batched`` is not a bool, or ``batch_size`` is not an int or None.
        ValueError: If ``batch_size`` is below 1, or set while ``batched`` is ``False``.

    Examples:
        >>> from hgp_lib.utils.validation import validate_batching
        >>> validate_batching(True, 64)  # no error
        >>> validate_batching(False, 64)
        Traceback (most recent call last):
        ...
        ValueError: batch_size requires batched=True
    """
    check_isinstance(batched, BOOL_TYPES, "batched")
    if batch_size is None:
        return
    if isinstance(batch_size, BOOL_TYPES) or not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be an int or None, is {type(batch_size)}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, is {batch_size}")
    if not batched:
        raise ValueError("batch_size requires batched=True")


def validate_num_literals(num_literals: int):
    """
    Validate ``num_literals`` parameter.

    Args:
        num_literals (int): Number of literals (must be > 1).

    Raises:
        TypeError: If not an integer.
        ValueError: If <= 1.

    Examples:
        >>> from hgp_lib.utils.validation import validate_num_literals
        >>> validate_num_literals(5)  # no error
        >>> validate_num_literals(1)
        Traceback (most recent call last):
        ...
        ValueError: Number of literals must be greater than 1, is '1'
    """
    check_isinstance(num_literals, int)
    if num_literals <= 1:
        raise ValueError(
            f"Number of literals must be greater than 1, is '{num_literals}'"
        )


def validate_operator_types(operator_types: Sequence[type[Rule]]):
    """
    Validate ``operator_types`` parameter.

    Args:
        operator_types (Sequence[type[Rule]]): Sequence of Rule subclasses.

    Raises:
        TypeError: If not a sequence or contains non-Rule types.
        ValueError: If fewer than 2 types.

    Examples:
        >>> from hgp_lib.rules import And, Or
        >>> from hgp_lib.utils.validation import validate_operator_types
        >>> validate_operator_types((And, Or))  # no error
        >>> validate_operator_types((And,))
        Traceback (most recent call last):
        ...
        ValueError: operator_types must have at least two operator types
    """
    check_isinstance(operator_types, Sequence)
    if len(operator_types) < 2:
        raise ValueError("operator_types must have at least two operator types")
    for operator_type in operator_types:
        if not issubclass(operator_type, Rule):
            raise TypeError(
                f"All operator types must be subclassing Rule. Found '{type(operator_type)}'"
            )


_BINARIZE_HINT = (
    "For raw data, use BooleanRuleClassifier or GPBenchmarker, which binarize it "
    "with their fitted binarizer."
)


def check_X(X: np.ndarray) -> None:
    """
    Validate binarized data: a 2-D boolean NumPy array.

    Only the type, the dtype and the number of dimensions are checked, so the cost does
    not depend on the size of the data.

    Args:
        X (np.ndarray): The binarized data, rows are instances and columns are features.

    Raises:
        TypeError: If ``X`` is not a NumPy array, or its dtype is not boolean.
        ValueError: If ``X`` is not 2-D.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.utils.validation import check_X
        >>> check_X(np.array([[True, False]]))  # no error
        >>> check_X(np.array([[1, 0]]))
        Traceback (most recent call last):
        ...
        TypeError: data must be a binarized boolean array, has dtype int64. For raw data, use BooleanRuleClassifier or GPBenchmarker, which binarize it with their fitted binarizer.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(
            f"data must be a binarized boolean NumPy array, is {type(X).__name__}. "
            + _BINARIZE_HINT
        )
    if not np.issubdtype(X.dtype, np.bool_):
        raise TypeError(
            f"data must be a binarized boolean array, has dtype {X.dtype}. "
            + _BINARIZE_HINT
        )
    if X.ndim != 2:
        raise ValueError(f"data must be a 2-D array, has shape {X.shape}")


def to_bool_labels(y: np.ndarray) -> np.ndarray:
    """
    Return binary labels as a boolean array.

    Boolean labels are returned as they are. Integer labels must only contain 0 and 1,
    and are converted once. Any other dtype, for example float labels, is rejected.

    Args:
        y (np.ndarray): 1-D binary labels.

    Returns:
        np.ndarray: The labels, with a boolean dtype.

    Raises:
        TypeError: If ``y`` is not a NumPy array, or has a dtype other than bool or
            integer.
        ValueError: If integer labels contain values other than 0 and 1.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.utils.validation import to_bool_labels
        >>> to_bool_labels(np.array([1, 0, 1])).tolist()
        [True, False, True]
        >>> to_bool_labels(np.array([0.0, 1.0]))
        Traceback (most recent call last):
        ...
        TypeError: labels must be boolean, or integers 0 and 1, has dtype float64
    """
    check_isinstance(y, np.ndarray, "labels")
    if np.issubdtype(y.dtype, np.bool_):
        return y
    if not np.issubdtype(y.dtype, np.integer):
        raise TypeError(
            f"labels must be boolean, or integers 0 and 1, has dtype {y.dtype}"
        )
    if not ((y == 0) | (y == 1)).all():
        raise ValueError("integer labels must only contain 0 and 1")
    return y.astype(bool)


def check_X_y(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    x_type: type[np.ndarray] | type[pd.DataFrame] = np.ndarray,
) -> np.ndarray:
    """
    Validate input data and labels, and return the labels as a boolean array.

    Checks that X is an instance of `x_type`, y is a numpy array, both are non-None,
    non-empty, 2-D/1-D respectively, and have the same number of samples. A NumPy ``X``
    must be binarized (boolean dtype, see `check_X`); a DataFrame ``X`` holds raw
    features and may have any dtypes. The labels are checked and converted by
    `to_bool_labels`.

    Args:
        X (np.ndarray | pd.DataFrame): Input data.
        y (np.ndarray): Target labels (1-D), boolean or integers 0 and 1.
        x_type (type[np.ndarray] | type[pd.DataFrame]): Expected type for X.
            Default: `np.ndarray`.

    Returns:
        np.ndarray: The labels, with a boolean dtype.

    Raises:
        ValueError: If X or y is None, empty, or have mismatched lengths, or the
            labels contain values other than 0 and 1.
        TypeError: If X is not an instance of `x_type`, a NumPy ``X`` is not boolean,
            or y is not an ndarray of booleans or integers.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.utils.validation import check_X_y
        >>> check_X_y(np.array([[True], [False]]), np.array([1, 0])).tolist()
        [True, False]
    """
    if X is None:
        raise ValueError("X (data) cannot be None")
    if y is None:
        raise ValueError("y (labels) cannot be None")

    check_isinstance(X, x_type, "X")
    check_isinstance(y, np.ndarray, "y")

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
    if isinstance(X, np.ndarray):
        check_X(X)
    return to_bool_labels(y)
