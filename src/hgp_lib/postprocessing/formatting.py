"""
Write a tree as a string over the original columns.

The tree is turned into a `Rule` whose literals refer to the text of its conditions,
and `Rule.to_str` writes it, so the notation is the one of rules: ``And(...)``,
``Or(...)`` and ``~``. A condition on a column is written as:

- numeric: ``x < 2.5``, ``1.5 <= x < 4.0``, ``2.5 <= x``, one interval per run of
  adjacent bins;
- categorical: ``color=red``, ``color in {red, blue}``;
- boolean: ``flag``, ``~flag``;
- missing values: ``x is NA``, ``x is not NA``;
- constants: ``True``, ``False``.

Every comparison is false for a missing value; only ``x is NA`` matches it. ``~`` is
plain negation, so ``~1.5 <= x < 4.0`` is true for a missing value. A condition is
written in the shorter of its positive form (an ``Or`` of the pieces it allows) and its
negative form (an ``And`` of the negated pieces it excludes), counting pieces and then
listed categories, and in its positive form on a tie.
"""

from ..preprocessing.features import BinFeature, BooleanFeature, CategoryFeature
from ..rules import Literal, Rule
from .domain import AND, OR, Column, Domain
from .tree import Cond, Const, Node, operator_rule


def to_str(
    node: Node, domain: Domain, indent: int = -1, term: str | None = None
) -> str:
    """
    Write ``node`` over the original columns.

    Args:
        node (Node): The tree to write.
        domain (Domain): The domain of its conditions.
        indent (int): The indentation level, as in `Rule.to_str`. ``-1`` writes one
            line. Default: `-1`.
        term (str | None): For a normal form, the operator of its terms (`AND` for a
            DNF, `OR` for a CNF). A condition is then only spliced into a term, so that
            each term of the tree stays one term of the string. Default: `None`.
    """
    if isinstance(node, Const):
        return str(node.value)
    texts: list[str] = []
    (rule,) = _display(node, domain, None, term, texts)
    return rule.to_str(texts, indent)


def _display(
    node: Node, domain: Domain, parent: str | None, term: str | None, texts: list
) -> list[Rule]:
    """
    The rules that write ``node``, whose literals are indices into ``texts``. A rule of
    the parent's kind is spliced into the parent.
    """
    if isinstance(node, Cond):
        operator, pieces, negated = condition(domain.columns[node.column], node.mask)
        literals = []
        for text in pieces:
            texts.append(text)
            literals.append(Literal(value=len(texts) - 1, negated=negated))
        if len(literals) == 1 or (operator == parent and term in (None, operator)):
            return literals
        return [operator_rule(operator, literals)]
    children = []
    for child in node.children:
        children.extend(_display(child, domain, node.operator, term, texts))
    if node.operator == parent:
        return children
    return [operator_rule(node.operator, children)]


def condition(column: Column, mask: int) -> tuple[str, list[str], bool]:
    """
    Write the condition ``mask`` (neither empty nor full) on ``column``.

    Returns:
        tuple[str, list[str], bool]: The operator joining the pieces, the pieces, and
            whether each piece is negated: `OR` and ``False`` for the positive form
            (the pieces ``mask`` allows), `AND` and ``True`` for the negative form (the
            pieces it excludes).
    """
    positive = _pieces(column, mask)
    negative = _pieces(column, column.full & ~mask)
    if _cost(negative) < _cost(positive):
        return AND, [text for text, _ in negative], True
    return OR, [text for text, _ in positive], False


def cost(column: Column, mask: int, parent: str | None = None) -> tuple[int, int, int]:
    """
    How long the condition ``mask`` is to read: its number of pieces, then the number of
    categories it lists, then whether it is negated.
    """
    positive = _cost(_pieces(column, mask))
    negative = _cost(_pieces(column, column.full & ~mask))
    if negative < positive:
        return (*negative, 1)
    return (*positive, 0)


def _cost(pieces: list) -> tuple[int, int]:
    return len(pieces), sum(listed for _, listed in pieces)


def _pieces(column: Column, mask: int) -> list[tuple[str, int]]:
    """
    The pieces whose ``Or`` is the non-empty mask ``mask``, each with the number of
    categories it lists.
    """
    label = column.label
    pieces = []
    values = mask & ~column.missing_bit
    if values == column.full & ~column.missing_bit:
        pieces.append((f"{label} is not NA", 1))
    elif values:
        cells = column.cells(values)
        if column.kind is BinFeature:
            for first, last in _runs(cells):
                left, right = column.values[first][0], column.values[last][1]
                pieces.append((str(BinFeature(label, left, right)), 1))
        elif column.kind is CategoryFeature:
            categories = [column.values[cell] for cell in cells]
            if len(categories) == 1:
                pieces.append((str(CategoryFeature(label, categories[0])), 1))
            else:
                listed = ", ".join(f"{category}" for category in categories)
                pieces.append((f"{label} in {{{listed}}}", len(categories)))
        elif column.kind is BooleanFeature:
            if cells == [0]:
                pieces.append((str(BooleanFeature(label)), 1))
            elif column.has_missing:
                pieces.append((f"{label}=False", 1))
            else:
                pieces.append((f"~{label}", 1))
    if mask & column.missing_bit:
        pieces.append((f"{label} is NA", 1))
    return pieces


def _runs(cells: list[int]) -> list[tuple[int, int]]:
    """The first and last cell of each run of consecutive cells."""
    runs = []
    for cell in cells:
        if runs and runs[-1][1] == cell - 1:
            runs[-1] = (runs[-1][0], cell)
        else:
            runs.append((cell, cell))
    return runs
