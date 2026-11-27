"""
Descriptions of the boolean features a binarizer produces.

A binarizer turns each original column into one or more boolean features. A
`FeatureSpec` records what one feature tests on its original column, so code that
reads rules (for example `hgp_lib.postprocessing`) does not have to parse feature
names. ``str(spec)`` is the feature name the binarizer gives it, before any suffix
added to keep names unique.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from math import inf


@dataclass(frozen=True)
class FeatureSpec:
    """
    Base class of the feature descriptions.

    Attributes:
        column (Hashable): The label of the original column the feature tests.
    """

    column: Hashable


@dataclass(frozen=True)
class BooleanFeature(FeatureSpec):
    """
    A boolean column passed through unchanged. True where the column is true.

    `Binarizer.get_feature_specs` also uses it for features it knows nothing about,
    with the feature name as the column.

    Examples:
        >>> from hgp_lib.preprocessing import BooleanFeature
        >>> str(BooleanFeature("smoker"))
        'smoker'
    """

    def __str__(self) -> str:
        return f"{self.column}"


@dataclass(frozen=True)
class CategoryFeature(FeatureSpec):
    """
    One category of a one-hot encoded column. True where the column equals ``value``.

    Attributes:
        column (Hashable): The label of the original column.
        value (Hashable): The category.

    Examples:
        >>> from hgp_lib.preprocessing import CategoryFeature
        >>> str(CategoryFeature("color", "red"))
        'color=red'
    """

    value: Hashable

    def __str__(self) -> str:
        return f"{self.column}={self.value}"


@dataclass(frozen=True)
class BinFeature(FeatureSpec):
    """
    One bin of a numeric column. True where ``left <= value < right``.

    The first bin of a column starts at ``-inf`` and the last one ends at ``inf``, so the
    bins of a column cover every number.

    Attributes:
        column (Hashable): The label of the original column.
        left (float): The left edge, included. ``-inf`` for the first bin.
        right (float): The right edge, excluded. ``inf`` for the last bin.

    Examples:
        >>> from math import inf
        >>> from hgp_lib.preprocessing import BinFeature
        >>> str(BinFeature("x", -inf, 3.0))
        'x < 3.0'
        >>> str(BinFeature("x", 1.0, 3.0))
        '1.0 <= x < 3.0'
        >>> str(BinFeature("x", 1.0, inf))
        '1.0 <= x'
    """

    left: float
    right: float

    def __str__(self) -> str:
        if self.left == -inf:
            return f"{self.column} < {self.right}"
        if self.right == inf:
            return f"{self.left} <= {self.column}"
        return f"{self.left} <= {self.column} < {self.right}"


@dataclass(frozen=True)
class MissingFeature(FeatureSpec):
    """
    The missing-value indicator of a column. True where the column's value is missing.

    The other features of the column are false for a missing value.

    Examples:
        >>> from hgp_lib.preprocessing import MissingFeature
        >>> str(MissingFeature("income"))
        'income_is_NA'
    """

    def __str__(self) -> str:
        return f"{self.column}_is_NA"
