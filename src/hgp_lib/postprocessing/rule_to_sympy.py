# rule_to_sympy.py

from __future__ import annotations

import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import sympy as sp

from .simplifier import ThresholdSimplifier

# Supported feature forms:
#
#   feature_name
#   feature_name=4
#   feature_name < 5.2
#   3.1 <= feature_name
#   2.7 <= feature_name < 6
#
# feature names themselves may contain spaces

_RE_EXACT = re.compile(
    r"^(?P<name>.+?)=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?)$"
)

_RE_UPPER = re.compile(
    r"^(?P<name>.+?)\s*<\s*"
    r"(?P<upper>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)

_RE_LOWER = re.compile(
    r"^(?P<lower>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*<=\s*(?P<name>.+)$"
)

_RE_RANGE = re.compile(
    r"^(?P<lower>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*<=\s*(?P<name>.+?)\s*<\s*"
    r"(?P<upper>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)


def _number(value: str) -> sp.Number:
    """
    Convert a decimal string to an exact SymPy number.

    Using Rational instead of Python float avoids introducing binary
    floating-point artifacts into the generated expression.
    """
    return sp.Float(value)


def _safe_symbol_name(name: str) -> str:
    """
    Convert a feature name into a valid, deterministic SymPy symbol name.

    Example:
        "mean radius" -> "mean_radius"
    """
    result = re.sub(r"\W+", "_", name.strip())
    result = result.strip("_")

    if not result:
        raise ValueError(f"Invalid feature name: {name!r}")

    if result[0].isdigit():
        result = "_" + result

    return result


def _feature_base_name(feature: str) -> str:
    """
    Extract the underlying column name from one of the supported
    relational feature descriptions.
    """
    feature = feature.strip()

    match = _RE_RANGE.match(feature)
    if match:
        return match.group("name").strip()

    match = _RE_EXACT.match(feature)
    if match:
        return match.group("name").strip()

    match = _RE_UPPER.match(feature)
    if match:
        return match.group("name").strip()

    match = _RE_LOWER.match(feature)
    if match:
        return match.group("name").strip()

    return feature


def feature_to_sympy(
    feature: str,
    symbols: dict[str, sp.Symbol],
) -> sp.Expr:
    """
    Convert one feature description into a SymPy Boolean expression.

    Examples
    --------
    "mean radius"
        -> mean_radius

    "mean radius=4"
        -> Eq(mean_radius, 4)

    "mean radius < 13.095"
        -> mean_radius < 13.095

    "13.095 <= mean radius"
        -> mean_radius >= 13.095

    "13.095 <= mean radius < 14.660"
        -> (mean_radius >= 13.095) & (mean_radius < 14.660)
    """
    feature = feature.strip()

    # most specific pattern first
    match = _RE_RANGE.match(feature)
    if match:
        name = match.group("name").strip()
        lower = _number(match.group("lower"))
        upper = _number(match.group("upper"))

        symbol = symbols[_feature_base_name(name)]

        return sp.And(
            symbol >= lower,
            symbol < upper,
        )

    match = _RE_EXACT.match(feature)
    if match:
        name = match.group("name").strip()
        value = _number(match.group("value"))

        symbol = symbols[_feature_base_name(name)]

        return sp.Eq(symbol, value)

    match = _RE_UPPER.match(feature)
    if match:
        name = match.group("name").strip()
        upper = _number(match.group("upper"))

        symbol = symbols[_feature_base_name(name)]

        return symbol < upper

    match = _RE_LOWER.match(feature)
    if match:
        lower = _number(match.group("lower"))
        name = match.group("name").strip()

        symbol = symbols[_feature_base_name(name)]

        return symbol >= lower

    name = feature
    return symbols[_feature_base_name(name)]


def _convert_rule(
    rule: Any,
    features: list[str],
    symbols: dict[str, sp.Symbol],
) -> sp.Expr:
    """
    Recursively convert the custom Rule/And/Or tree to SymPy.
    """
    kind = getattr(rule, "kind", rule.__class__.__name__)
    negated = bool(getattr(rule, "negated", False))

    if kind == "Literal":
        value = getattr(rule, "value", None)

        if value is None:
            raise ValueError("Literal has no value.")

        if not isinstance(value, int) and not isinstance(value, np.int64):
            raise TypeError(
                f"Literal.value must be an int feature index, got {value!r}"
            )

        if value < 0 or value >= len(features):
            raise IndexError(
                f"Literal feature index {value} is outside features[0:{len(features)}]."
            )

        expression = feature_to_sympy(features[value], symbols)

    elif kind == "And":
        children = getattr(rule, "subrules", None) or []

        if not children:
            expression = sp.true
        else:
            parsed_children = [
                _convert_rule(child, features, symbols) for child in children
            ]
            expression = sp.And(*parsed_children)

    elif kind == "Or":
        children = getattr(rule, "subrules", None) or []

        if not children:
            expression = sp.false
        else:
            parsed_children = [
                _convert_rule(child, features, symbols) for child in children
            ]
            expression = sp.Or(*parsed_children)

    else:
        raise TypeError(
            f"Unsupported rule node type {kind!r}. Expected Literal, And, or Or."
        )

    if negated:
        expression = sp.Not(expression)

    return expression


def rule_to_sympy(
    clf: Any,
    *,
    prepare_rule: Callable[[sp.Expr], sp.Expr] | None = None,
    simplify: bool | None = True,
    simplify_rule: Callable[[sp.Expr], sp.Expr] | None = None,
) -> Any:
    """
    WARNING: May take a long time to execute.

    Convert a rule to a SymPy expression and save all intermediate
    representations on the parent object.

    Parameters
    ----------
    clf:
        Object containing at least:
            - clf.rule
            - clf.feature_names

    prepare_rule:
        Optional function applied to the SymPy expression before
        simplify_rule(). Disabled by default.

    simplify:
        Simplify the function before returning, enabled by default.

    simplify_rule:
        Main simplification function.

        Defaults to:
            ThresholdSimplifier().simplify()

    Returns
    -------
    clf
        The same input object, with intermediate results attached.
    """

    original_rule = clf.rule

    features = getattr(clf, "features_names", None)

    if features is None:
        features = getattr(clf, "feature_names", None)

    if features is None:
        raise AttributeError(
            "clf must contain either 'features_names' or 'feature_names'."
        )

    rule_features = list(features)

    # The actual underlying column names may be available through:
    #
    #   clf.binarizer._original_columns
    #
    # but the feature descriptions themselves are sufficient to construct
    # the symbols. Using the original columns, when available, also gives
    # a stable symbol namespace.

    original_columns = None

    binarizer = getattr(clf, "binarizer", None)

    if binarizer is not None:
        original_columns = getattr(
            binarizer,
            "_original_columns",
            None,
        )

    if original_columns is None:
        original_columns = [_feature_base_name(feature) for feature in features]

    original_columns = list(dict.fromkeys(original_columns))

    symbols = {
        column: sp.Symbol(_safe_symbol_name(column), real=True)
        for column in original_columns
    }

    formatted_rule = None

    if hasattr(clf, "format_rule") and callable(clf.format_rule):
        formatted_rule = clf.format_rule()
    else:
        formatted_rule = None

    sympy_rule = _convert_rule(
        original_rule,
        features,
        symbols,
    )

    sympy_rule_string = str(sympy_rule)

    return_obj = {
        "original_rule": original_rule,
        "rule_features": rule_features,
        "original_columns": original_columns,
        "sympy_symbols": symbols,
        "formatted_rule": formatted_rule,
        "sympy_rule": sympy_rule,
        "sympy_rule_string": sympy_rule_string,
    }

    if prepare_rule is not None:
        sympy_rule = prepare_rule(sympy_rule)
        return_obj["prepared_rule"] = sympy_rule
        return_obj["prepared_rule_string"] = str(sympy_rule)

    if simplify_rule is None:
        simplify_rule = ThresholdSimplifier().simplify

    if simplify:
        sympy_rule = simplify_rule(sympy_rule)
        return_obj["simplified_rule"] = SimpleNamespace(**sympy_rule)

    clf.sympy = SimpleNamespace(**return_obj)

    return clf
