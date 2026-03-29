import inspect
from typing import Sequence, Tuple, Type, Callable, Any
import numpy as np
import pandas as pd

from ..rules import Rule


class ComplexityCheck:
    """
    Create a validity predicate that rejects rules exceeding ``max_complexity`` nodes.

    Intended for use as the ``check_valid`` argument of ``BooleanGPConfig``.

    Args:
        max_complexity (int):
            Maximum allowed node count. Default: `100`.

    Examples:
    >>> from hgp_lib.rules import Literal, And
    >>> from hgp_lib.utils.validation import ComplexityCheck
    >>> check = ComplexityCheck(3)
    >>> check(Literal(value=0))
    True
    >>> check(And([Literal(value=0), Literal(value=1)]))
    True
    >>> check(And([Literal(value=0), And([Literal(value=1), Literal(value=2)])]))
    False
    """

    def __init__(self, max_complexity: int = 100):
        self.max_complexity = max_complexity

    def __call__(self, rule: Rule) -> bool:
        """
        Check if rule complexity (node count) is within a limit.

        Args:
            rule (Rule): The rule to check.

        Returns:
            bool: ``True`` if ``len(rule) <= self.max_complexity``.

        Examples:
            >>> from hgp_lib.rules import Literal, And
            >>> from hgp_lib.utils.validation import ComplexityCheck
            >>> ComplexityCheck(5)(Literal(value=0))
            True
            >>> ComplexityCheck(2)(And([Literal(value=0), Literal(value=1)]))
            False
        """
        return len(rule) <= self.max_complexity


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


def check_isinstance(value: Any, expected_type: Type | Tuple[Type, ...]):
    """
    Check that a value is an instance of expected type(s).

    Args:
        value (Any): Value to check.
        expected_type (Type | Tuple[Type, ...]): Expected type or tuple of types.

    Raises:
        TypeError: If value is not an instance of expected type.
    """
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


def validate_operator_types(operator_types: Sequence[Type[Rule]]):
    """
    Validate ``operator_types`` parameter.

    Args:
        operator_types (Sequence[Type[Rule]]): Sequence of Rule subclasses.

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


def check_X_y(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    x_type: Type[np.ndarray] | Type[pd.DataFrame] = np.ndarray,
):
    """
    Validate input data and labels.

    Checks that X is an instance of `x_type`, y is a numpy array, both are
    non-None, non-empty, 2-D/1-D respectively, and have the same number of
    samples.

    Args:
        X (np.ndarray | pd.DataFrame): Input data.
        y (np.ndarray): Target labels (1-D).
        x_type (Type[np.ndarray] | Type[pd.DataFrame]): Expected type for X.
            Default: `np.ndarray`.

    Raises:
        ValueError: If X or y is None, empty, or have mismatched lengths.
        TypeError: If X is not an instance of `x_type` or y is not an ndarray.
    """
    if X is None:
        raise ValueError("X (data) cannot be None")
    if y is None:
        raise ValueError("y (labels) cannot be None")

    check_isinstance(X, x_type)
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
