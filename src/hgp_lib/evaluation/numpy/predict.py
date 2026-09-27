"""
Rule evaluation on NumPy boolean arrays.

Two algorithms with identical results:

- `evaluate` does as few operations as possible. Literal children of an operator are
  gathered with one fancy index and reduced along one axis, at the cost of a temporary
  block with one column per literal.
- `evaluate_low_memory` updates a single result buffer in place. Literal children are
  folded in with ufuncs writing to that buffer, so they allocate nothing.

Operators never return a view of ``data``, so their results can be updated in place.
A literal on its own returns a column view, which callers only read.
"""

import numpy as np
from numpy import ndarray

from ...rules import Rule
from ...rules.operators import And, Or


def _is_and(rule: Rule) -> bool:
    if isinstance(rule, And):
        return True
    if isinstance(rule, Or):
        return False
    raise TypeError(
        f"Unsupported rule type: {type(rule).__name__}. Backends evaluate Literal, And and Or."
    )


def evaluate(rule: Rule, data: ndarray) -> ndarray:
    """
    Evaluate a rule with the fewest operations.

    Args:
        rule (Rule): The rule to evaluate.
        data (ndarray): 2-D boolean array, rows are instances and columns are features.

    Returns:
        ndarray: 1-D boolean predictions. A single literal returns a view of ``data``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.numpy.predict import evaluate
        >>> from hgp_lib.rules import And, Literal, Or
        >>> data = np.array([[True, False, True], [False, False, True], [True, True, False]])
        >>> evaluate(And([Literal(value=0), Or([Literal(value=1), Literal(value=2)])]), data)
        array([ True, False,  True])
    """
    value = rule.value
    if value is not None:
        column = data[:, value]
        return ~column if rule.negated else column

    is_and = _is_and(rule)
    columns = []
    negated = []
    operators = []
    for subrule in rule.subrules:
        if subrule.value is not None:
            columns.append(subrule.value)
            negated.append(subrule.negated)
        else:
            operators.append(subrule)

    if len(columns) == 1:
        column = data[:, columns[0]]
        mask = ~column if negated[0] else column.copy()
    elif columns:
        block = data[:, columns]  # fancy indexing copies, so XOR can run in place
        if any(negated):
            block ^= np.array(negated)
        mask = block.all(axis=1) if is_and else block.any(axis=1)
    else:
        mask = evaluate(operators.pop(), data)

    if is_and:
        for operator in operators:
            mask &= evaluate(operator, data)
    else:
        for operator in operators:
            mask |= evaluate(operator, data)

    if rule.negated:
        np.logical_not(mask, out=mask)
    return mask


def evaluate_low_memory(rule: Rule, data: ndarray) -> ndarray:
    """
    Evaluate a rule with as little temporary memory as possible.

    Args:
        rule (Rule): The rule to evaluate.
        data (ndarray): 2-D boolean array, rows are instances and columns are features.

    Returns:
        ndarray: 1-D boolean predictions. A single literal returns a view of ``data``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.numpy.predict import evaluate_low_memory
        >>> from hgp_lib.rules import And, Literal, Or
        >>> data = np.array([[True, False, True], [False, False, True], [True, True, False]])
        >>> evaluate_low_memory(Or([Literal(value=0, negated=True), Literal(value=1)]), data)
        array([False,  True,  True])
    """
    value = rule.value
    if value is not None:
        column = data[:, value]
        return ~column if rule.negated else column

    is_and = _is_and(rule)
    first, *rest = rule.subrules
    if first.value is not None:
        column = data[:, first.value]
        mask = ~column if first.negated else column.copy()
    else:
        mask = evaluate_low_memory(first, data)

    for subrule in rest:
        value = subrule.value
        if value is not None:
            column = data[:, value]
            # With booleans, ``a & ~b`` is ``a > b`` and ``a | ~b`` is ``a >= b``.
            if is_and:
                ufunc = np.greater if subrule.negated else np.logical_and
            else:
                ufunc = np.greater_equal if subrule.negated else np.logical_or
            ufunc(mask, column, out=mask)
        elif is_and:
            mask &= evaluate_low_memory(subrule, data)
        else:
            mask |= evaluate_low_memory(subrule, data)

    if rule.negated:
        np.logical_not(mask, out=mask)
    return mask
