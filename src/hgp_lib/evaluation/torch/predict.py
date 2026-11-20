"""
Rule evaluation on PyTorch boolean tensors: the implementations of
`hgp_lib.evaluation.numpy.predict`, with the torch API.

``data`` has the shape (rows, features). The backend stores it column-major, so a
literal reads one contiguous column.
"""

# TODO: On accelerators, each literal costs one kernel launch. Evaluating a whole
#  population level by level (all literals of one depth gathered and reduced by segment)
#  would need a few launches per level instead.

import torch
from torch import Tensor

from ...rules import Rule
from ...rules.operators import And, Or
from ..numpy.predict import unsupported_rule


def evaluate_block(rule: Rule, data: Tensor) -> Tensor:
    """
    Evaluate a rule, gathering the literal children of each operator into one block.

    Args:
        rule (Rule): The rule to evaluate.
        data (Tensor): 2-D boolean tensor, rows are instances and columns are features.

    Returns:
        Tensor: 1-D boolean predictions, on the device of ``data``. A single literal
            returns a view of ``data``.

    Examples:
        >>> import torch
        >>> from hgp_lib.evaluation.torch.predict import evaluate_block
        >>> from hgp_lib.rules import And, Literal, Or
        >>> data = torch.tensor([[True, False, True], [False, False, True], [True, True, False]])
        >>> evaluate_block(And([Literal(value=0), Or([Literal(value=1), Literal(value=2)])]), data)
        tensor([ True, False,  True])
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
        mask = data[:, columns]  # advanced indexing copies, so XOR can run in place
        if any_negated:
            mask ^= torch.tensor(negated, device=data.device)
        mask = mask.all(dim=1) if is_and else mask.any(dim=1)
    else:
        mask = evaluate_block(operators.pop(), data)

    if is_and:
        for subrule in operators:
            mask &= evaluate_block(subrule, data)
    else:
        for subrule in operators:
            mask |= evaluate_block(subrule, data)

    if rule.negated:
        torch.logical_not(mask, out=mask)
    return mask


def evaluate_inplace(rule: Rule, data: Tensor) -> Tensor:
    """
    Evaluate a rule, updating one result buffer per operator in place.

    Args:
        rule (Rule): The rule to evaluate.
        data (Tensor): 2-D boolean tensor, rows are instances and columns are features.

    Returns:
        Tensor: 1-D boolean predictions, on the device of ``data``. A single literal
            returns a view of ``data``.

    Examples:
        >>> import torch
        >>> from hgp_lib.evaluation.torch.predict import evaluate_inplace
        >>> from hgp_lib.rules import Literal, Or
        >>> data = torch.tensor([[True, False, True], [False, False, True], [True, True, False]])
        >>> evaluate_inplace(Or([Literal(value=0, negated=True), Literal(value=1)]), data)
        tensor([False,  True,  True])
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
        mask = ~column if first.negated else column.clone()
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
        torch.logical_not(mask, out=mask)
    return mask


def evaluate_fused(rule: Rule, data: Tensor) -> Tensor:
    """
    Evaluate a rule like `evaluate_inplace`, fusing negated literals into the update.

    For booleans, ``a & ~b`` is ``a > b`` and ``a | ~b`` is ``a >= b``.

    Args:
        rule (Rule): The rule to evaluate.
        data (Tensor): 2-D boolean tensor, rows are instances and columns are features.

    Returns:
        Tensor: 1-D boolean predictions, on the device of ``data``. A single literal
            returns a view of ``data``.

    Examples:
        >>> import torch
        >>> from hgp_lib.evaluation.torch.predict import evaluate_fused
        >>> from hgp_lib.rules import Literal, Or
        >>> data = torch.tensor([[True, False, True], [False, False, True], [True, True, False]])
        >>> evaluate_fused(Or([Literal(value=0, negated=True), Literal(value=1)]), data)
        tensor([False,  True,  True])
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
        mask = ~column if first.negated else column.clone()
    else:
        mask = evaluate_fused(first, data)

    if is_and:
        for subrule in subrules:
            if subrule.value is not None:
                if subrule.negated:
                    torch.gt(mask, data[:, subrule.value], out=mask)
                else:
                    mask &= data[:, subrule.value]
            else:
                mask &= evaluate_fused(subrule, data)
    else:
        for subrule in subrules:
            if subrule.value is not None:
                if subrule.negated:
                    torch.ge(mask, data[:, subrule.value], out=mask)
                else:
                    mask |= data[:, subrule.value]
            else:
                mask |= evaluate_fused(subrule, data)

    if rule.negated:
        torch.logical_not(mask, out=mask)
    return mask


EVALUATIONS = {
    "block": evaluate_block,
    "inplace": evaluate_inplace,
    "fused": evaluate_fused,
}
"""The evaluation implementations, by the name ``TorchBackend(evaluation=...)`` takes."""
