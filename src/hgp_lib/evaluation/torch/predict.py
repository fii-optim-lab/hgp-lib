"""
Rule evaluation on PyTorch boolean tensors.

``columns`` holds one row per feature (the transposed data), so a literal reads one
contiguous row. Each operator updates a single result buffer in place, one tensor
operation per literal child. On the benchmark rules this was faster, on the CPU and on
MPS, than gathering the literal children into a block, which needs an index tensor per
operator.

Operators never return a view of ``columns``, so their results can be updated in place.
A literal on its own returns a view, which callers only read.
"""

# TODO: On accelerators, the time is the launch cost of one operation per literal.
#  Evaluating a whole population level by level (all literals of one depth gathered and
#  reduced by segment) would need a few operations per level instead. Not planned for
#  2.0.0.

import torch
from torch import Tensor

from ...rules import Rule
from ..numpy.predict import _is_and


def evaluate(rule: Rule, columns: Tensor) -> Tensor:
    """
    Evaluate a rule on boolean feature columns.

    Args:
        rule (Rule): The rule to evaluate.
        columns (Tensor): 2-D boolean tensor, one row per feature and one column per
            instance.

    Returns:
        Tensor: 1-D boolean predictions, on the device of ``columns``. A single literal
            returns a view of ``columns``.

    Examples:
        >>> import torch
        >>> from hgp_lib.evaluation.torch.predict import evaluate
        >>> from hgp_lib.rules import And, Literal, Or
        >>> columns = torch.tensor([[True, False, True], [False, False, True], [True, True, False]])
        >>> evaluate(And([Literal(value=0), Or([Literal(value=1), Literal(value=2)])]), columns)
        tensor([ True, False,  True])
    """
    value = rule.value
    if value is not None:
        column = columns[value]
        return ~column if rule.negated else column

    is_and = _is_and(rule)
    first, *rest = rule.subrules
    if first.value is not None:
        column = columns[first.value]
        mask = ~column if first.negated else column.clone()
    else:
        mask = evaluate(first, columns)

    for subrule in rest:
        value = subrule.value
        if value is not None:
            # With booleans, ``a & ~b`` is ``a > b`` and ``a | ~b`` is ``a >= b``.
            if is_and:
                operation = torch.gt if subrule.negated else torch.logical_and
            else:
                operation = torch.ge if subrule.negated else torch.logical_or
            operation(mask, columns[value], out=mask)
        elif is_and:
            mask &= evaluate(subrule, columns)
        else:
            mask |= evaluate(subrule, columns)

    if rule.negated:
        torch.logical_not(mask, out=mask)
    return mask
