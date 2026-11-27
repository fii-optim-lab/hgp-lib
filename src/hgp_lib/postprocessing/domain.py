"""
The possible values of each original column, and the values each feature allows.

Postprocessing reasons about the original columns, not about the binarized features.
Each column has a small, fixed set of possible values, its *cells*:

- a numeric column has one cell per bin;
- a categorical column has one cell per category seen at fit;
- a boolean column has a true and a false cell;
- a column with a ``MissingFeature`` also has a missing cell, after the others.

Every feature allows a subset of the cells of one column, stored as an integer
bitmask, so ``And``, ``Or`` and ``~`` become intersection, union and complement.
"""

from collections.abc import Sequence
from itertools import pairwise
from math import inf

from ..preprocessing.base import Binarizer
from ..preprocessing.features import (
    BinFeature,
    BooleanFeature,
    CategoryFeature,
    FeatureSpec,
    MissingFeature,
)
from ..rules import Literal, Rule

AND = "and"
OR = "or"

_VALUE_KINDS = (BinFeature, CategoryFeature, BooleanFeature)


class Column:
    """
    The cells of one original column.

    Attributes:
        label (Hashable): The column label.
        kind (type | None): `BinFeature`, `CategoryFeature` or `BooleanFeature`, or
            ``None`` for a column that only has a missing-value indicator.
        values (tuple): One entry per value cell: ``(left, right)`` for a bin, the
            category for a category, ``True`` and ``False`` for a boolean column, and
            ``None`` for the single "present" cell of a column with only a
            missing-value indicator.
        has_missing (bool): Whether the column has a missing cell, after the values.
        features (tuple[int | None, ...]): For each cell, the index of a feature that
            is true exactly on that cell, or ``None`` when no feature is.
        full (int): The mask of all cells.
        missing_bit (int): The mask of the missing cell, or 0.
    """

    __slots__ = (
        "features",
        "full",
        "has_missing",
        "kind",
        "label",
        "missing_bit",
        "values",
    )

    def __init__(self, label, kind, values: tuple, has_missing: bool, features: tuple):
        self.label = label
        self.kind = kind
        self.values = values
        self.has_missing = has_missing
        self.features = features
        self.full = (1 << len(features)) - 1
        self.missing_bit = 1 << len(values) if has_missing else 0

    def cells(self, mask: int) -> list[int]:
        """The cell indices in ``mask``, in order."""
        return [i for i in range(len(self.features)) if mask >> i & 1]

    def writings(self, mask: int, parent: str | None) -> list[tuple[int, str, list]]:
        """
        The ways to write ``mask`` with this column's features, as a child of a
        ``parent`` operator (`AND`, `OR` or ``None``).

        ``mask`` is the ``Or`` of the features of its cells (positive writing), or the
        ``And`` of the negated features of the other cells (negative writing). A writing
        is missing when a cell it needs has no feature.

        Returns:
            list[tuple[int, str, list[tuple[int, bool]]]]: ``(cost, kind, literals)``
                entries, the positive writing first. ``cost`` counts rule nodes: an
                ``Or`` is spliced into an ``Or`` parent and an ``And`` into an ``And``
                parent, so it needs no operator node there. ``kind`` is ``"literal"``
                for a single literal, otherwise the operator joining the
                ``(feature, negated)`` literals.
        """
        result = []
        for cells, negated, operator in (
            (self.cells(mask), False, OR),
            (self.cells(self.full & ~mask), True, AND),
        ):
            features = [self.features[i] for i in cells]
            if None in features:
                continue
            literals = [(f, negated) for f in features]
            if len(literals) == 1:
                result.append((1, "literal", literals))
            else:
                result.append(
                    (len(literals) + (operator != parent), operator, literals)
                )
        return result

    def cost(self, mask: int, parent: str | None) -> int:
        """The fewest rule nodes that write ``mask`` as a child of ``parent``."""
        return min(cost for cost, _, _ in self.writings(mask, parent))


class Domain:
    """
    The columns of a set of features, and the cells each feature allows.

    Attributes:
        columns (list[Column]): The columns, in the order of their first feature.
        literals (list[tuple[int, int]]): For each feature index, the index of its
            column and the mask of the cells it allows.
        names (list[str] | None): The feature names, or ``None`` when the features were
            not given (then literals are shown by index, as in ``Rule.to_str``).
    """

    def __init__(self, specs: Sequence[FeatureSpec], names: list[str] | None):
        self.names = names
        self.columns: list[Column] = []
        self.literals: list[tuple[int, int]] = []

        groups: dict = {}
        for index, spec in enumerate(specs):
            groups.setdefault(spec.column, []).append((index, spec))

        cell_of: dict[int, tuple[int, int]] = {}
        for label, members in groups.items():
            column, cells = _build_column(label, members)
            for index, cell in cells.items():
                cell_of[index] = (len(self.columns), 1 << cell)
            self.columns.append(column)
        self.literals = [cell_of[index] for index in range(len(specs))]

    @classmethod
    def of(cls, features, rule: Rule) -> "Domain":
        """
        Build the domain of ``features``, checking that it covers ``rule``.

        Args:
            features: A fitted `Binarizer`, a sequence of `FeatureSpec` (strings are
                read as `BooleanFeature`), or ``None`` to treat each literal of
                ``rule`` as an independent boolean feature named by its index.
            rule (Rule): The rule to postprocess.

        Raises:
            TypeError: If ``features`` has an unsupported type.
            ValueError: If the features are inconsistent, or a literal of ``rule`` has
                no feature.
        """
        values = [node.value for node in rule.flatten() if isinstance(node, Literal)]
        if features is None:
            count = max(values) + 1 if values else 0
            domain = cls([BooleanFeature(i) for i in range(count)], None)
        elif isinstance(features, Binarizer):
            specs = features.get_feature_specs()
            domain = cls(specs, features.get_feature_names_out())
        elif isinstance(features, Sequence) and not isinstance(features, str):
            specs = [_as_spec(item) for item in features]
            domain = cls(specs, [str(spec) for spec in specs])
        else:
            raise TypeError(
                "features should be a fitted Binarizer, a sequence of FeatureSpec or "
                f"feature names, or None, but is {type(features)}"
            )
        for value in values:
            if not 0 <= value < len(domain.literals):
                raise ValueError(
                    f"Literal {value} has no feature: there are "
                    f"{len(domain.literals)} features."
                )
        return domain


def _as_spec(item) -> FeatureSpec:
    if isinstance(item, FeatureSpec):
        return item
    if isinstance(item, str):
        return BooleanFeature(item)
    raise TypeError(
        f"features should contain FeatureSpec or str items, but contains {type(item)}"
    )


def _build_column(label, members) -> tuple[Column, dict[int, int]]:
    """The column of ``members`` (index, spec) pairs, and the cell of each index."""
    for _, spec in members:
        if not isinstance(spec, (*_VALUE_KINDS, MissingFeature)):
            raise TypeError(f"Unsupported feature spec: {type(spec).__name__}")
    value_specs = [(i, s) for i, s in members if not isinstance(s, MissingFeature)]
    kinds = list(dict.fromkeys(type(s) for _, s in value_specs))
    if len(kinds) > 1:
        names = ", ".join(kind.__name__ for kind in kinds)
        raise ValueError(f"Column {label!r} has features of different kinds: {names}")
    kind = kinds[0] if kinds else None

    if kind is BinFeature:
        unique = sorted({s for _, s in value_specs}, key=lambda s: s.left)
        _check_bins(label, unique)
        values = tuple((s.left, s.right) for s in unique)
        cell_of_spec = {s: c for c, s in enumerate(unique)}
    elif kind is CategoryFeature:
        unique = list(dict.fromkeys(s for _, s in value_specs))
        values = tuple(s.value for s in unique)
        cell_of_spec = {s: c for c, s in enumerate(unique)}
    elif kind is BooleanFeature:
        values = (True, False)
        cell_of_spec = {value_specs[0][1]: 0}
    else:
        values = (None,)
        cell_of_spec = {}

    has_missing = len(value_specs) < len(members)
    missing_cell = len(values)
    features: list = [None] * (len(values) + has_missing)
    cells = {}
    for index, spec in members:
        cell = missing_cell if isinstance(spec, MissingFeature) else cell_of_spec[spec]
        cells[index] = cell
        if features[cell] is None:
            features[cell] = index
    return Column(label, kind, values, has_missing, tuple(features)), cells


def _check_bins(label, bins: list[BinFeature]) -> None:
    edges_ok = bins[0].left == -inf and bins[-1].right == inf
    edges_ok = edges_ok and all(a.right == b.left for a, b in pairwise(bins))
    if not edges_ok or any(not b.left < b.right for b in bins):
        raise ValueError(
            f"The bins of column {label!r} must cover every number once, from -inf "
            f"to inf: {[str(b) for b in bins]}"
        )
