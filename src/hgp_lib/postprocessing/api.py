from collections.abc import Sequence
from enum import Enum

from ..preprocessing.base import Binarizer
from ..preprocessing.features import FeatureSpec
from ..rules import And, Literal, Or, Rule
from ..utils.validation import check_isinstance
from .domain import AND, OR, Domain
from .formatting import cost as text_cost
from .formatting import to_str
from .normal_form import canonical as _canonical
from .normal_form import mechanical
from .optimize import Cost, rule_cost
from .optimize import optimize as _optimize
from .tree import Const, Node, constant_rule, from_rule, to_rule

Features = Binarizer | Sequence[FeatureSpec | str] | None


class CanonicalForm(str, Enum):
    """
    The normal forms `postprocess` can produce.

    Examples:
        >>> from hgp_lib.postprocessing import CanonicalForm
        >>> CanonicalForm("dnf") is CanonicalForm.DNF
        True
    """

    DNF = "dnf"
    """Disjunctive normal form: an ``Or`` of ``And`` terms."""
    CNF = "cnf"
    """Conjunctive normal form: an ``And`` of ``Or`` clauses."""


def postprocess(
    rule: Rule,
    features: Features = None,
    *,
    optimize: bool = True,
    canonical: CanonicalForm | str | None = None,
) -> Rule:
    """
    Return a simplified rule that makes the same predictions as ``rule``.

    The result is a new rule over the same features, so it can replace ``rule``
    anywhere, for example in `hgp_lib.evaluation.predict`. ``rule`` is not modified.

    ``features`` tells which features come from the same original column. That is what
    makes, for example, ``And(x < 1, 1 <= x < 2)`` always false: both are bins of
    ``x``. Without it, every feature is treated as an independent boolean, which is
    still exact but simplifies less.

    With ``optimize=True`` (the default):

    - Negations are pushed down to the literals, and nested operators of the same kind
      are flattened.
    - The literals on the same column inside an operator are merged, and written back
      with the fewest literals. A rule can only use the existing features, so merging
      never creates a new bin edge: ``Or(1 <= x < 2, 2 <= x < 3, 3 <= x < 4)`` becomes
      ``And(~x < 1, ~4 <= x)`` when these are all the bins of ``x``. Use
      `postprocess_to_str` to read the merged bins as one interval.
    - A literal that the rest of the rule already decides is replaced by that
      constant: in ``And(a, Or(a, b))`` the ``Or`` always holds. Constants, repeated
      parts and absorbed parts are removed.
    - The result never has more nodes than ``rule``. A rule that is always true or
      always false is returned as ``Or(l, ~l)`` or ``And(l, ~l)``, with ``l`` its first
      literal, since a rule has no constant node; a rule of at most 3 nodes is then
      returned as is.

    With ``canonical="dnf"`` or ``"cnf"``, the result is a disjunctive or conjunctive
    normal form:

    - With ``optimize=True``, it is the canonical form: the prime terms (or clauses)
      that are needed, chosen in a fixed order. It depends only on what the rule
      computes, so equivalent rules give the same result. It is not always the smallest
      normal form, since finding that is NP-hard. A constant is ``Or(0, ~0)`` or
      ``And(0, ~0)``, over feature 0.
    - With ``optimize=False``, the rule is converted mechanically: negations pushed
      down, operators flattened, and ``And`` distributed over ``Or`` (DNF) or ``Or`` over
      ``And`` (CNF). Nothing is removed.

    A normal form can be much larger than the rule, and computing it can take long: an
    ``And`` of ``k`` ``Or`` nodes with ``m`` literals each has ``m**k`` DNF terms. See
    the postprocessing guide for the sizes that are fast.

    The results are exact under the assumptions of the binarizer: a column has missing
    values only if it had some at fit (``UnseenNaNWarning``), and only the categories
    seen at fit (``UnseenCategoryWarning``). A row that breaks them can be predicted
    differently by the postprocessed rule.

    Args:
        rule (Rule): A rule of `Literal`, `And` and `Or` nodes.
        features (Binarizer | Sequence[FeatureSpec | str] | None): The features the
            literals refer to: a fitted `Binarizer`, its `FeatureSpec` list (see
            `Binarizer.get_feature_specs`), or feature names, read as independent
            boolean features. ``None`` treats every literal as an independent boolean
            feature. Default: `None`.
        optimize (bool): Whether to simplify the rule. Default: `True`.
        canonical (CanonicalForm | str | None): ``None`` keeps the structure of the
            rule; ``"dnf"`` or ``"cnf"`` (case-insensitive, or a `CanonicalForm`) returns
            that normal form. Default: `None`.

    Returns:
        Rule: The postprocessed rule.

    Raises:
        TypeError: If ``rule`` contains a node other than `Literal`, `And` and `Or`, or
            an argument has the wrong type.
        ValueError: If ``canonical`` is not a known form, a literal has no feature, or
            the features of a column are inconsistent.
        NotFittedError: If ``features`` is a binarizer that is not fitted.

    Examples:
        The bins of a numeric column ``age`` and the categories of ``color``:

        >>> from math import inf
        >>> from hgp_lib.postprocessing import postprocess
        >>> from hgp_lib.preprocessing import BinFeature, CategoryFeature
        >>> from hgp_lib.rules import And, Literal, Or
        >>> features = [
        ...     BinFeature("age", -inf, 30.0),
        ...     BinFeature("age", 30.0, 50.0),
        ...     BinFeature("age", 50.0, inf),
        ...     CategoryFeature("color", "red"),
        ...     CategoryFeature("color", "green"),
        ... ]
        >>> names = [str(feature) for feature in features]
        >>> rule = And([
        ...     Or([Literal(value=0), Literal(value=1)]),
        ...     Or([Literal(value=0), Literal(value=3)]),
        ...     Literal(value=2, negated=True),
        ... ])
        >>> rule.to_str(names)
        'And(Or(age < 30.0, 30.0 <= age < 50.0), Or(age < 30.0, color=red), ~50.0 <= age)'

        ``~50.0 <= age`` already makes the first ``Or`` true:

        >>> postprocess(rule, features).to_str(names)
        'And(Or(age < 30.0, color=red), ~50.0 <= age)'
        >>> postprocess(rule, features, canonical="dnf").to_str(names)
        'Or(age < 30.0, And(~50.0 <= age, color=red))'

        Without the features, the literals are independent, so only the repeated
        literal goes:

        >>> postprocess(rule).to_str(names)
        'And(Or(age < 30.0, 30.0 <= age < 50.0), Or(age < 30.0, color=red), ~50.0 <= age)'
        >>> postprocess(And([Literal(value=0), Literal(value=0)])).to_str(names)
        'age < 30.0'
    """
    form = _check(rule, optimize, canonical)
    domain = Domain.of(features, rule)
    if not optimize:
        if form is None:
            return _copy(rule)
        return mechanical(rule, form.value)
    node = _postprocess(rule, domain, form, rule_cost)
    if isinstance(node, Const):
        if form is not None:
            return constant_rule(node.value, 0)
        if len(rule) <= 3:
            return _copy(rule)
        return constant_rule(node.value, _first_literal(rule).value)
    return to_rule(node, domain, _term(form))


def postprocess_to_str(
    rule: Rule,
    features: Features = None,
    *,
    optimize: bool = True,
    canonical: CanonicalForm | str | None = None,
    indent: int = -1,
) -> str:
    """
    Postprocess ``rule`` like `postprocess`, and write the result over the original
    columns.

    A string can say what a rule cannot: with ``optimize=True``, the literals on one
    column inside an operator are written as one condition (adjacent bins as one
    interval, several categories as a set), and a constant rule is ``True`` or
    ``False``. The notation is that of `Rule.to_str`:

    - numeric: ``x < 2.5``, ``1.5 <= x < 4.0``, ``2.5 <= x``;
    - categorical: ``color=red``, ``color in {red, blue}``;
    - boolean: ``flag``, ``~flag``;
    - missing values: ``x is NA``, ``x is not NA``;
    - constants: ``True``, ``False``.

    Every comparison is false for a missing value; only ``x is NA`` matches it. ``~`` is
    plain negation, so ``~1.5 <= x < 4.0`` is true for a missing value. A condition is
    written in the shorter of its positive form (an ``Or`` of the pieces it allows) and
    its negative form (an ``And`` of the negated pieces it excludes), counting pieces
    and then listed categories, and in its positive form on a tie.

    With ``optimize=False``, the literals are written with the feature names, as
    ``rule.to_str(feature_names)`` does.

    Args:
        rule (Rule): A rule of `Literal`, `And` and `Or` nodes.
        features (Binarizer | Sequence[FeatureSpec | str] | None): As in `postprocess`.
            Default: `None`.
        optimize (bool): As in `postprocess`. Default: `True`.
        canonical (CanonicalForm | str | None): As in `postprocess`. Default: `None`.
        indent (int): The indentation level, as in `Rule.to_str`. ``-1`` writes one
            line, ``0`` one node per line. Default: `-1`.

    Returns:
        str: The postprocessed rule.

    Raises:
        TypeError: As in `postprocess`, or if ``indent`` is not an integer.
        ValueError: As in `postprocess`, or if ``indent`` is less than -1.
        NotFittedError: If ``features`` is a binarizer that is not fitted.

    Examples:
        >>> from math import inf
        >>> from hgp_lib.postprocessing import postprocess_to_str
        >>> from hgp_lib.preprocessing import BinFeature, CategoryFeature
        >>> from hgp_lib.rules import And, Literal, Or
        >>> features = [
        ...     BinFeature("age", -inf, 30.0),
        ...     BinFeature("age", 30.0, 50.0),
        ...     BinFeature("age", 50.0, inf),
        ...     CategoryFeature("color", "red"),
        ...     CategoryFeature("color", "green"),
        ...     CategoryFeature("color", "blue"),
        ...     CategoryFeature("color", "black"),
        ... ]
        >>> rule = And([
        ...     Or([Literal(value=0), Literal(value=1)]),
        ...     Or([Literal(value=3), Literal(value=4)]),
        ... ])
        >>> postprocess_to_str(rule, features)
        'And(age < 50.0, color in {red, green})'
        >>> postprocess_to_str(rule, features, optimize=False)
        'And(Or(age < 30.0, 30.0 <= age < 50.0), Or(color=red, color=green))'
        >>> print(postprocess_to_str(rule, features, indent=0).expandtabs(4))
        And(
            age < 50.0,
            color in {red, green}
        )

        A rule that is always true:

        >>> postprocess_to_str(Or([Literal(value=3), Literal(value=3, negated=True)]))
        'True'
    """
    form = _check(rule, optimize, canonical)
    check_isinstance(indent, int, "indent")
    if indent < -1:
        raise ValueError(f"indent should be -1 or more, but is {indent}")
    domain = Domain.of(features, rule)
    if not optimize:
        result = rule if form is None else mechanical(rule, form.value)
        return result.to_str(domain.names, indent)
    node = _postprocess(rule, domain, form, text_cost)
    return to_str(node, domain, indent, _term(form))


def _postprocess(
    rule: Rule, domain: Domain, form: "CanonicalForm | None", cost: Cost
) -> Node:
    node = _optimize(from_rule(rule, domain), domain, cost)
    if form is not None:
        node = _canonical(node, domain, form.value)
    return node


def _term(form: "CanonicalForm | None") -> str | None:
    """The operator of the terms of a normal form."""
    if form is None:
        return None
    return AND if form is CanonicalForm.DNF else OR


def _check(rule, optimize, canonical) -> CanonicalForm | None:
    check_isinstance(rule, Rule, "rule")
    check_isinstance(optimize, bool, "optimize")
    for node in rule.flatten():
        if not isinstance(node, (Literal, And, Or)):
            raise TypeError(
                f"Unsupported rule type: {type(node).__name__}. Postprocessing "
                "supports Literal, And and Or."
            )
    if canonical is None or isinstance(canonical, CanonicalForm):
        return canonical
    check_isinstance(canonical, str, "canonical")
    try:
        return CanonicalForm(canonical.lower())
    except ValueError:
        raise ValueError(
            f"canonical should be None, 'dnf' or 'cnf', but is {canonical!r}"
        ) from None


def _copy(rule: Rule) -> Rule:
    result = rule.copy()
    result.parent = None
    return result


def _first_literal(rule: Rule) -> Literal:
    while not isinstance(rule, Literal):
        rule = rule.subrules[0]
    return rule
