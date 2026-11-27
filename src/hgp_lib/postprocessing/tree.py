"""
The expression tree postprocessing works on.

A rule is read into a tree of `Cond` leaves (one column, a mask of allowed cells), `Op`
nodes (``And`` / ``Or``) and the constants `TRUE` and `FALSE`, with the negations
pushed down to the leaves. The tree is immutable. Each node has a ``key`` that is equal
for equal nodes, whatever the order of the children.
"""

from ..rules import And, Literal, Or, Rule
from .domain import AND, OR, Domain


class Const:
    """The constant `TRUE` or `FALSE`."""

    __slots__ = ("key", "value")

    def __init__(self, value: bool):
        self.value = value
        self.key = ("const", value)

    def __repr__(self) -> str:
        return str(self.value)


TRUE = Const(True)
FALSE = Const(False)


class Cond:
    """The cells ``mask`` of column ``column`` (a column index of the domain)."""

    __slots__ = ("column", "key", "mask")

    def __init__(self, column: int, mask: int):
        self.column = column
        self.mask = mask
        self.key = ("cond", column, mask)

    def __repr__(self) -> str:
        return f"Cond({self.column}, {self.mask:b})"


class Op:
    """An ``And`` (``operator == AND``) or ``Or`` of at least two children."""

    __slots__ = ("children", "key", "operator")

    def __init__(self, operator: str, children: tuple):
        self.operator = operator
        self.children = children
        self.key = (operator, frozenset(child.key for child in children))

    def __repr__(self) -> str:
        return f"{self.operator.capitalize()}{self.children}"


Node = Const | Cond | Op


def other(operator: str) -> str:
    """The dual operator."""
    return OR if operator == AND else AND


def identity(operator: str) -> Const:
    """The constant an operator ignores: `TRUE` for ``And``, `FALSE` for ``Or``."""
    return TRUE if operator == AND else FALSE


def make_cond(domain: Domain, column: int, mask: int) -> Node:
    """A condition, or a constant when it allows no cell or every cell."""
    if mask == 0:
        return FALSE
    if mask == domain.columns[column].full:
        return TRUE
    return Cond(column, mask)


def make_op(operator: str, children: list) -> Node:
    """An operator over ``children``; one child is returned as is, none as identity."""
    if not children:
        return identity(operator)
    if len(children) == 1:
        return children[0]
    return Op(operator, tuple(children))


def from_rule(rule: Rule, domain: Domain, negate: bool = False) -> Node:
    """
    Read ``rule`` into a tree, pushing its negations down to the conditions.

    Raises:
        TypeError: If the rule contains a type other than `Literal`, `And` and `Or`.
    """
    negated = rule.negated != negate
    if isinstance(rule, Literal):
        column, mask = domain.literals[rule.value]
        if negated:
            mask = domain.columns[column].full & ~mask
        return make_cond(domain, column, mask)
    if isinstance(rule, And):
        operator = OR if negated else AND
    elif isinstance(rule, Or):
        operator = AND if negated else OR
    else:
        raise TypeError(
            f"Unsupported rule type: {type(rule).__name__}. Postprocessing supports "
            "Literal, And and Or."
        )
    children = [from_rule(child, domain, negated) for child in rule.subrules]
    return make_op(operator, children)


def negation(node: Node, domain: Domain) -> Node:
    """The negation of ``node``, with the negations pushed down to the conditions."""
    if isinstance(node, Const):
        return FALSE if node.value else TRUE
    if isinstance(node, Cond):
        full = domain.columns[node.column].full
        return Cond(node.column, full & ~node.mask)
    children = tuple(negation(child, domain) for child in node.children)
    return Op(other(node.operator), children)


def to_rule(node: Node, domain: Domain, term: str | None = None) -> Rule:
    """
    Write a non-constant tree as a rule over the domain's features.

    Each condition is written with the fewest nodes (see `Column.writings`). For a
    normal form, ``term`` is the operator of its terms (`AND` for a DNF, `OR` for a
    CNF): a condition is then written as one literal or as a ``term`` operator, when
    possible, so that each term of the tree stays one term of the rule.
    """
    (rule,) = _emit(node, domain, None, term)
    return rule


def _emit(node: Node, domain: Domain, parent: str | None, term: str | None) -> list:
    if isinstance(node, Cond):
        writings = domain.columns[node.column].writings(node.mask, parent)
        if term is not None:
            writings = [w for w in writings if w[1] in ("literal", term)] or writings
        _, kind, literals = min(writings, key=lambda writing: writing[0])
        rules = [Literal(value=f, negated=negated) for f, negated in literals]
        if kind in ("literal", parent):
            return rules
        return [operator_rule(kind, rules)]
    if isinstance(node, Op):
        rules = []
        for child in node.children:
            rules.extend(_emit(child, domain, node.operator, term))
        if node.operator == parent:
            return rules
        return [operator_rule(node.operator, rules)]
    raise ValueError(f"A constant has no rule: {node}")


def operator_rule(operator: str, subrules: list) -> Rule:
    """An `And` (``operator == AND``) or `Or` rule that takes ``subrules`` without copying."""
    cls = And if operator == AND else Or
    return cls(subrules=subrules, copy_subrules=False)


def constant_rule(value: bool, feature: int) -> Rule:
    """
    A rule that is always ``value``: ``Or(l, ~l)`` for true, ``And(l, ~l)`` for false,
    with ``l`` the literal of ``feature``.
    """
    literals = [Literal(value=feature), Literal(value=feature, negated=True)]
    return operator_rule(OR if value else AND, literals)
