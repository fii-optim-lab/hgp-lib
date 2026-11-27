"""
Simplify a tree without changing what it computes (``optimize=True``).

The simplification repeats these steps until nothing changes:

- nested operators of the same kind are flattened, and an operator with one child is
  replaced by it;
- the conditions on the same column inside an operator are merged: their cells are
  intersected in an ``And`` and united in an ``Or``;
- a condition is checked against what its context already decides: inside an ``And``,
  the other children only matter where its conditions hold, and inside an ``Or``, where
  they do not. A condition that the context decides becomes `TRUE` or `FALSE`, and one
  that the context makes cheaper to write is replaced. What "cheaper" means depends on
  the output: fewer rule nodes for a rule (`Column.cost`), and a shorter text for a
  string (`formatting.cost`);
- constants are folded, repeated children are removed, and ``Or(A, And(A, B))`` and
  ``And(A, Or(A, B))`` become ``A``.
"""

from collections.abc import Callable

from .domain import AND, OR, Column, Domain
from .tree import FALSE, TRUE, Cond, Const, Node, Op, identity, make_op, other

Cost = Callable[[Column, int, "str | None"], object]


def rule_cost(column: Column, mask: int, parent: str | None) -> int:
    """The number of rule nodes that write ``mask`` as a child of ``parent``."""
    return column.cost(mask, parent)


def optimize(node: Node, domain: Domain, cost: Cost = rule_cost) -> Node:
    """
    Simplify ``node`` until it no longer changes.

    Args:
        node (Node): A tree read with `from_rule`.
        domain (Domain): The domain of its conditions.
        cost (Callable): How expensive a condition is to write, given its column, its
            mask and the operator of its parent. It chooses between conditions that are
            equivalent in their context. Default: `rule_cost`.

    Returns:
        Node: An equivalent tree, possibly a constant.
    """
    while True:
        result = _simplify(node, {}, None, domain, cost)
        if result.key == node.key:
            return result
        node = result


def _simplify(
    node: Node, context: dict, parent: str | None, domain: Domain, cost: Cost
) -> Node:
    """
    Simplify ``node``, knowing that each column ``c`` in ``context`` only takes the
    cells ``context[c]`` where ``node`` matters.
    """
    if isinstance(node, Const):
        return node
    if isinstance(node, Cond):
        return _restrict(node.column, node.mask, context, parent, domain, cost)
    return _simplify_op(node.operator, node.children, context, domain, cost)


def _restrict(
    column: int,
    mask: int,
    context: dict,
    parent: str | None,
    domain: Domain,
    cost: Cost,
) -> Node:
    """
    The condition ``mask`` on ``column`` where the column takes ``context[column]``.

    Outside the context, the cells do not matter, so any mask with the same cells inside
    it works. The cheapest of ``mask`` (preferred on a tie), its cells inside the
    context, and its cells plus those outside the context is kept.
    """
    cells = domain.columns[column]
    allowed = context.get(column, cells.full)
    effective = mask & allowed
    if effective == 0:
        return FALSE
    if effective == allowed:
        return TRUE
    candidates = (mask, effective, mask | (cells.full & ~allowed))
    return Cond(column, min(candidates, key=lambda m: cost(cells, m, parent)))


def _simplify_op(
    operator: str, children: tuple, context: dict, domain: Domain, cost: Cost
) -> Node:
    absorbing = identity(other(operator))
    items = list(children)
    while True:
        masks: dict[int, int] = {}
        order: list = []  # A column index for its merged condition, otherwise a node.
        for child in _flatten(items, operator):
            if child is absorbing:
                return absorbing
            if isinstance(child, Const):
                continue
            if isinstance(child, Cond):
                if child.column not in masks:
                    masks[child.column] = child.mask
                    order.append(child.column)
                elif operator == AND:
                    masks[child.column] &= child.mask
                else:
                    masks[child.column] |= child.mask
            else:
                order.append(child)

        # The conditions, and what they tell the other children.
        conditions = {}
        inner = dict(context)
        for column, mask in masks.items():
            condition = _restrict(column, mask, context, operator, domain, cost)
            if condition is absorbing:
                return absorbing
            conditions[column] = condition
            full = domain.columns[column].full
            allowed = context.get(column, full)
            inner[column] = allowed & (mask if operator == AND else full & ~mask)

        simplified = []
        again = False
        for item in order:
            if isinstance(item, int):
                simplified.append(conditions[item])
                continue
            child = _simplify(item, inner, operator, domain, cost)
            # A child that became a constant, a condition or an operator of the same
            # kind is merged into this operator in another round.
            if isinstance(child, (Const, Cond)) or child.operator == operator:
                again = True
            simplified.append(child)
        items = simplified
        if not again:
            break

    kept = []
    keys = set()
    for child in items:
        if not isinstance(child, Const) and child.key not in keys:
            keys.add(child.key)
            kept.append(child)

    # Or(A, And(A, B)) is A, and And(A, Or(A, B)) is A.
    for child in list(kept):
        if isinstance(child, Op) and any(
            _absorbs(sibling, child, operator)
            for sibling in kept
            if sibling is not child
        ):
            kept.remove(child)
    return make_op(operator, kept)


def _absorbs(sibling: Node, child: Op, operator: str) -> bool:
    """
    Whether ``sibling`` makes ``child`` redundant in an ``operator`` node: in an ``Or``,
    when ``child`` implies ``sibling``; in an ``And``, when ``sibling`` implies
    ``child``. Checked part by part: each part of the implied side must follow from one
    part of the other side.
    """
    inner = other(operator)
    child_parts = child.children
    sibling_parts = (
        sibling.children
        if isinstance(sibling, Op) and sibling.operator == inner
        else (sibling,)
    )
    if operator == OR:  # child = And(...) implies sibling = And(...)
        return all(any(_implies(c, s) for c in child_parts) for s in sibling_parts)
    # sibling = Or(...) implies child = Or(...)
    return all(any(_implies(s, c) for c in child_parts) for s in sibling_parts)


def _implies(a: Node, b: Node) -> bool:
    if a.key == b.key:
        return True
    return (
        isinstance(a, Cond)
        and isinstance(b, Cond)
        and a.column == b.column
        and a.mask & ~b.mask == 0
    )


def _flatten(items, operator: str):
    for item in items:
        if isinstance(item, Op) and item.operator == operator:
            yield from _flatten(item.children, operator)
        else:
            yield item
