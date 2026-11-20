"""
Rule evaluation on NumPy boolean arrays.

Three implementations with the same predictions, selected by
``NumpyBackend(evaluation=...)``:

- `evaluate_block` (``"block"``) gathers the literal children of an operator into one
  block with a single fancy index, applies their negations with one XOR, and reduces the
  block with ``all`` or ``any``. It needs few NumPy calls, and a temporary block with
  one column per literal.
- `evaluate_inplace` (``"inplace"``) keeps one result buffer per operator and updates it
  in place with each child. A negated literal child is negated into a temporary first.
- `evaluate_fused` (``"fused"``) is `evaluate_inplace` with the negation fused into the
  update: for booleans, ``a & ~b`` is ``a > b`` and ``a | ~b`` is ``a >= b``, so a
  negated literal child takes one comparison ufunc writing into the buffer, without a
  temporary.

Operators never return a view of ``data``, so their results can be updated in place.
A literal on its own returns a column view, which callers only read.

Backends evaluate `Literal`, `And` and `Or`; any other rule type raises a ``TypeError``.
"""

import numpy as np
from numpy import ndarray

from ...rules import Rule
from ...rules.operators import And, Or


def unsupported_rule(rule: Rule) -> TypeError:
    """The error for a rule type the backends do not evaluate."""
    return TypeError(
        f"Unsupported rule type: {type(rule).__name__}. Backends evaluate Literal, And "
        "and Or."
    )


def evaluate_block(rule: Rule, data: ndarray) -> ndarray:
    """
    Evaluate a rule, gathering the literal children of each operator into one block.

    Args:
        rule (Rule): The rule to evaluate.
        data (ndarray): 2-D boolean array, rows are instances and columns are features.

    Returns:
        ndarray: 1-D boolean predictions. A single literal returns a view of ``data``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.numpy.predict import evaluate_block
        >>> from hgp_lib.rules import And, Literal, Or
        >>> data = np.array([[True, False, True], [False, False, True], [True, True, False]])
        >>> evaluate_block(And([Literal(value=0), Or([Literal(value=1), Literal(value=2)])]), data)
        array([ True, False,  True])
    """
    if rule.value is not None:
        return ~data[:, rule.value] if rule.negated else data[:, rule.value]

    if isinstance(rule, And):
        is_and = True
    elif isinstance(rule, Or):
        is_and = False
    else:
        raise unsupported_rule(rule)

    columns = []
    negated = []
    operators = []
    any_negated = False
    for subrule in rule.subrules:
        if subrule.value is not None:
            columns.append(subrule.value)
            if subrule.negated:
                negated.append(True)
                any_negated = True
            else:
                negated.append(False)
        else:
            operators.append(subrule)

    if columns:
        mask = data[:, columns]  # fancy indexing copies, so XOR can run in place
        if any_negated:
            mask ^= np.array(negated)
        mask = mask.all(axis=1) if is_and else mask.any(axis=1)
    else:
        mask = evaluate_block(operators.pop(), data)

    if is_and:
        for subrule in operators:
            mask &= evaluate_block(subrule, data)
    else:
        for subrule in operators:
            mask |= evaluate_block(subrule, data)

    if rule.negated:
        np.logical_not(mask, out=mask)
    return mask


def evaluate_inplace(rule: Rule, data: ndarray) -> ndarray:
    """
    Evaluate a rule, updating one result buffer per operator in place.

    Args:
        rule (Rule): The rule to evaluate.
        data (ndarray): 2-D boolean array, rows are instances and columns are features.

    Returns:
        ndarray: 1-D boolean predictions. A single literal returns a view of ``data``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.numpy.predict import evaluate_inplace
        >>> from hgp_lib.rules import Literal, Or
        >>> data = np.array([[True, False, True], [False, False, True], [True, True, False]])
        >>> evaluate_inplace(Or([Literal(value=0, negated=True), Literal(value=1)]), data)
        array([False,  True,  True])
    """
    if rule.value is not None:
        return ~data[:, rule.value] if rule.negated else data[:, rule.value]

    if isinstance(rule, And):
        is_and = True
    elif isinstance(rule, Or):
        is_and = False
    else:
        raise unsupported_rule(rule)

    subrules = iter(rule.subrules)
    first = next(subrules)
    if first.value is not None:
        column = data[:, first.value]
        mask = ~column if first.negated else column.copy()
    else:
        mask = evaluate_inplace(first, data)

    if is_and:
        for subrule in subrules:
            if subrule.value is not None:
                if subrule.negated:
                    mask &= ~data[:, subrule.value]
                else:
                    mask &= data[:, subrule.value]
            else:
                mask &= evaluate_inplace(subrule, data)
    else:
        for subrule in subrules:
            if subrule.value is not None:
                if subrule.negated:
                    mask |= ~data[:, subrule.value]
                else:
                    mask |= data[:, subrule.value]
            else:
                mask |= evaluate_inplace(subrule, data)

    if rule.negated:
        np.logical_not(mask, out=mask)
    return mask


def evaluate_fused(rule: Rule, data: ndarray) -> ndarray:
    """
    Evaluate a rule like `evaluate_inplace`, fusing negated literals into the update.

    Args:
        rule (Rule): The rule to evaluate.
        data (ndarray): 2-D boolean array, rows are instances and columns are features.

    Returns:
        ndarray: 1-D boolean predictions. A single literal returns a view of ``data``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.numpy.predict import evaluate_fused
        >>> from hgp_lib.rules import Literal, Or
        >>> data = np.array([[True, False, True], [False, False, True], [True, True, False]])
        >>> evaluate_fused(Or([Literal(value=0, negated=True), Literal(value=1)]), data)
        array([False,  True,  True])
    """
    if rule.value is not None:
        return ~data[:, rule.value] if rule.negated else data[:, rule.value]

    if isinstance(rule, And):
        is_and = True
    elif isinstance(rule, Or):
        is_and = False
    else:
        raise unsupported_rule(rule)

    subrules = iter(rule.subrules)
    first = next(subrules)
    if first.value is not None:
        column = data[:, first.value]
        mask = ~column if first.negated else column.copy()
    else:
        mask = evaluate_fused(first, data)

    if is_and:
        for subrule in subrules:
            if subrule.value is not None:
                if subrule.negated:
                    np.greater(mask, data[:, subrule.value], out=mask)
                else:
                    mask &= data[:, subrule.value]
            else:
                mask &= evaluate_fused(subrule, data)
    else:
        for subrule in subrules:
            if subrule.value is not None:
                if subrule.negated:
                    np.greater_equal(mask, data[:, subrule.value], out=mask)
                else:
                    mask |= data[:, subrule.value]
            else:
                mask |= evaluate_fused(subrule, data)

    if rule.negated:
        np.logical_not(mask, out=mask)
    return mask


EVALUATIONS = {
    "block": evaluate_block,
    "inplace": evaluate_inplace,
    "fused": evaluate_fused,
}
"""The evaluation implementations, by the name ``NumpyBackend(evaluation=...)`` takes."""
