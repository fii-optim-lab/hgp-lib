"""
Tests of `hgp_lib.postprocessing`, through its public API only.

Equivalence is checked exactly: the test data has one row for every combination of
the cells of its columns (one value per bin, each category, and a missing value where
the column has missing values), so two rules that predict the same on it predict the
same on every row the binarizer can produce. The string output is read back by a
small parser of its documented notation and evaluated on the raw rows.
"""

import itertools
import re
import unittest
import warnings
from math import inf

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from hgp_lib.evaluation import predict
from hgp_lib.postprocessing import CanonicalForm, postprocess, postprocess_to_str
from hgp_lib.preprocessing import (
    BinFeature,
    BooleanFeature,
    CategoryFeature,
    MissingFeature,
    StandardBinarizer,
)
from hgp_lib.rules import And, Literal, Or, Rule

MODES = [
    (optimize, canonical)
    for optimize in (True, False)
    for canonical in (None, "dnf", "cnf")
]


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def fit_binarizer(df: pd.DataFrame, num_bins: int = 3) -> StandardBinarizer:
    binarizer = StandardBinarizer(num_bins=num_bins, progress_bar=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        binarizer.fit_transform(df)
    return binarizer


def all_cells(binarizer: StandardBinarizer, fit: pd.DataFrame) -> pd.DataFrame:
    """One raw row per combination of the cells of the columns of ``fit``."""
    specs = binarizer.get_feature_specs()
    choices = []
    for column in fit.columns:
        own = [spec for spec in specs if spec.column == column]
        values = []
        for spec in own:
            if isinstance(spec, BinFeature):
                # The left edge, which the bin includes, or a value below the right one.
                values.append(spec.left if spec.left > -inf else spec.right - 1.0)
            elif isinstance(spec, CategoryFeature):
                values.append(spec.value)
            elif isinstance(spec, BooleanFeature):
                values.extend([True, False])
        if not values:  # Only a missing-value indicator: any present value.
            values.append(fit[column].dropna().iloc[0])
        if any(isinstance(spec, MissingFeature) for spec in own):
            values.append(None)
        choices.append(values)
    rows = pd.DataFrame(list(itertools.product(*choices)), columns=fit.columns)
    for column in fit.columns:
        if isinstance(fit[column].dtype, pd.CategoricalDtype):
            rows[column] = pd.Categorical(
                rows[column], categories=fit[column].cat.categories
            )
        elif fit[column].dtype == bool:
            rows[column] = rows[column].astype(bool)
        else:
            rows[column] = rows[column].astype(float)
    return rows


def mixed_data() -> pd.DataFrame:
    """A numeric column with and one without missing values, a categorical column
    with missing values, a boolean column, and an all-distinct column with missing
    values (dropped except for its indicator)."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, 7.0, 8.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            "color": pd.Categorical(["r", "g", "b", "r", None, "g", "b", "r", "g"]),
            "flag": [True, False, True, True, False, False, True, False, True],
            "id": pd.Categorical(["a", "b", "c", "d", "e", "f", "g", "h", None]),
        }
    )


# ---------------------------------------------------------------------------
# Reading the string output back
# ---------------------------------------------------------------------------

_NUMBER = r"[-+0-9.einf]+"
_ATOMS = [
    (r"(?P<c>\w+) is NA", lambda df, c: df[c].isna()),
    (r"(?P<c>\w+) is not NA", lambda df, c: df[c].notna()),
    (
        rf"(?P<left>{_NUMBER}) <= (?P<c>\w+) < (?P<right>{_NUMBER})",
        lambda df, c, left, right: (df[c] >= float(left)) & (df[c] < float(right)),
    ),
    (rf"(?P<left>{_NUMBER}) <= (?P<c>\w+)", lambda df, c, left: df[c] >= float(left)),
    (rf"(?P<c>\w+) < (?P<right>{_NUMBER})", lambda df, c, right: df[c] < float(right)),
    (
        r"(?P<c>\w+) in \{(?P<values>.*)\}",
        lambda df, c, values: df[c].isin(values.split(", ")),
    ),
    (r"(?P<c>\w+)=(?P<value>.+)", lambda df, c, value: _equals(df[c], value)),
    (r"(?P<c>\w+)", lambda df, c: df[c].astype(bool)),
]


def _equals(series: pd.Series, value: str) -> pd.Series:
    if series.dtype == bool:
        return series == (value == "True")
    return (series == value).fillna(False).astype(bool)


def evaluate_str(text: str, df: pd.DataFrame) -> np.ndarray:
    """Evaluate the documented string notation on raw rows."""
    result, rest = _parse(text.replace("\n", "").replace("\t", ""), df)
    assert rest == "", f"Unparsed text {rest!r} in {text!r}"
    return np.asarray(result, dtype=bool)


def _parse(text: str, df: pd.DataFrame):
    if text.startswith("~"):
        value, rest = _parse(text[1:], df)
        return ~value, rest
    for name, combine in (("And(", np.logical_and), ("Or(", np.logical_or)):
        if text.startswith(name):
            rest = text[len(name) :]
            values = []
            while True:
                value, rest = _parse(rest, df)
                values.append(value)
                if rest.startswith(", "):
                    rest = rest[2:]
                elif rest.startswith(","):
                    rest = rest[1:]
                else:
                    break
            assert rest.startswith(")"), rest
            return combine.reduce(np.array(values)), rest[1:]
    for constant in ("True", "False"):
        if text.startswith(constant) and not re.match(r"\w", text[len(constant) :]):
            return np.full(len(df), constant == "True"), text[len(constant) :]
    depth = 0
    for end, char in enumerate(text):
        depth += {"{": 1, "}": -1}.get(char, 0)
        if depth == 0 and char in ",)":
            break
    else:
        end = len(text)
    return _atom(text[:end], df), text[end:]


def _atom(text: str, df: pd.DataFrame) -> np.ndarray:
    for pattern, evaluate in _ATOMS:
        match = re.fullmatch(pattern, text)
        if match and match["c"] in df.columns:
            return np.asarray(evaluate(df, **match.groupdict()), dtype=bool)
    raise AssertionError(f"Unknown condition {text!r}")


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def random_rule(rng: np.random.Generator, num_features: int, depth: int) -> Rule:
    if depth == 0 or rng.random() < 0.3:
        return Literal(
            value=int(rng.integers(num_features)), negated=bool(rng.random() < 0.5)
        )
    operator = And if rng.random() < 0.5 else Or
    count = rng.integers(2, 4)
    children = [random_rule(rng, num_features, depth - 1) for _ in range(count)]
    return operator(children, negated=bool(rng.random() < 0.25), copy_subrules=False)


def rewrite(rule: Rule, rng: np.random.Generator) -> Rule:
    """An equivalent rule: shuffled children, De Morgan, regrouping, repetition."""
    if isinstance(rule, Literal):
        return rule.copy()
    children = [rewrite(child, rng) for child in rule.subrules]
    rng.shuffle(children)
    if rng.random() < 0.3:
        children.append(children[0].copy())
    if len(children) > 2 and rng.random() < 0.3:
        children = [children[0], type(rule)(children[1:])]
    result = type(rule)(children, negated=rule.negated, copy_subrules=False)
    if rng.random() < 0.3:
        # And(a, b) is ~Or(~a, ~b): swap the operator and negate it and its children.
        dual = Or if isinstance(rule, And) else And
        for child in children:
            child.negated = not child.negated
        result = dual(children, negated=not rule.negated, copy_subrules=False)
    return result


class Cells:
    """
    The cells of each column, read from the feature specs: the feature true on the
    cell, or ``("not", f)`` for the cell no feature is true on (the false value of a
    boolean column, or the present value of a column with only a missing indicator).
    """

    def __init__(self, specs: list, data: np.ndarray):
        self.column_of = [spec.column for spec in specs]
        self.cells: dict = {}
        for index, spec in enumerate(specs):
            self.cells.setdefault(spec.column, set()).add(index)
        for column, features in self.cells.items():
            kinds = {type(specs[f]) for f in features}
            if kinds <= {BooleanFeature, MissingFeature} and len(features) == 1:
                features.add(("not", next(iter(features))))
        # The cell of each row in each column.
        self.row_cells = {}
        for column, features in self.cells.items():
            own = [f for f in features if isinstance(f, int)]
            fallback = next((f for f in features if not isinstance(f, int)), None)
            values = []
            for row in data:
                true = [f for f in own if row[f]]
                values.append(true[0] if true else fallback)
            self.row_cells[column] = values

    def of_term(self, term: Rule) -> dict:
        """The cells a conjunction of literals allows, by column."""
        literals = term.subrules if isinstance(term, And) else [term]
        allowed = {column: set(cells) for column, cells in self.cells.items()}
        for literal in literals:
            column = self.column_of[literal.value]
            cell = {literal.value}
            if literal.negated:
                allowed[column] -= cell
            else:
                allowed[column] &= cell
        return allowed

    def widenings(self, allowed: dict):
        for column, cells in allowed.items():
            for cell in self.cells[column] - cells:
                yield column, cell

    def rows_in(self, allowed: dict) -> np.ndarray:
        result = np.ones(len(next(iter(self.row_cells.values()))), dtype=bool)
        for column, cells in allowed.items():
            result &= np.array([cell in cells for cell in self.row_cells[column]])
        return result


def literal_count(rule: Rule) -> int:
    return sum(isinstance(node, Literal) for node in rule.flatten())


def is_normal_form(rule: Rule, outer: type, inner: type) -> bool:
    """Whether ``rule`` is an ``outer`` of ``inner`` of literals, without negations."""
    if isinstance(rule, Literal):
        return True
    if rule.negated:
        return False
    if isinstance(rule, inner):
        return all(isinstance(child, Literal) for child in rule.subrules)
    return isinstance(rule, outer) and all(
        isinstance(child, Literal)
        or (
            isinstance(child, inner)
            and not child.negated
            and all(isinstance(grand, Literal) for grand in child.subrules)
        )
        for child in rule.subrules
    )


# ---------------------------------------------------------------------------
# Properties on random rules
# ---------------------------------------------------------------------------


class TestRandomRules(unittest.TestCase):
    """Random rules over mixed-type data, checked on every cell combination."""

    @classmethod
    def setUpClass(cls):
        cls.fit = mixed_data()
        cls.binarizer = fit_binarizer(cls.fit)
        cls.rows = all_cells(cls.binarizer, cls.fit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.data = cls.binarizer.transform(cls.rows).to_numpy()
        cls.num_features = len(cls.binarizer.get_feature_names_out())

    def rules(self, count: int, depth: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        return [random_rule(rng, self.num_features, depth) for _ in range(count)]

    def test_all_cell_combinations_are_present(self):
        # Every feature is true and false somewhere, and each row is a distinct cell
        # combination.
        self.assertTrue(self.data.any(axis=0).all())
        self.assertFalse(self.data.all(axis=0).any())
        self.assertEqual(len(np.unique(self.data, axis=0)), len(self.data))

    def test_same_predictions_in_every_mode(self):
        for i, rule in enumerate(self.rules(150, depth=4)):
            expected = predict(rule, self.data)
            for optimize, canonical in MODES:
                if not optimize and canonical is not None and len(rule) > 25:
                    continue  # The mechanical normal form can be very large.
                with self.subTest(rule=i, optimize=optimize, canonical=canonical):
                    result = postprocess(
                        rule, self.binarizer, optimize=optimize, canonical=canonical
                    )
                    np.testing.assert_array_equal(predict(result, self.data), expected)

    def test_strings_say_what_the_rule_does(self):
        for i, rule in enumerate(self.rules(150, depth=4)):
            expected = predict(rule, self.data)
            for optimize, canonical in MODES:
                if optimize is False:
                    continue  # Feature names, not conditions on the columns.
                with self.subTest(rule=i, canonical=canonical):
                    text = postprocess_to_str(
                        rule, self.binarizer, optimize=True, canonical=canonical
                    )
                    np.testing.assert_array_equal(
                        evaluate_str(text, self.rows), expected
                    )

    def test_optimized_rule_is_never_larger(self):
        for i, rule in enumerate(self.rules(300, depth=5)):
            with self.subTest(rule=i):
                self.assertLessEqual(len(postprocess(rule, self.binarizer)), len(rule))

    def test_optimizing_twice_changes_nothing(self):
        for i, rule in enumerate(self.rules(300, depth=5)):
            with self.subTest(rule=i):
                once = postprocess(rule, self.binarizer)
                self.assertEqual(str(postprocess(once, self.binarizer)), str(once))

    def test_normal_forms_have_their_shape(self):
        for i, rule in enumerate(self.rules(150, depth=4)):
            for optimize in (True, False):
                if not optimize and len(rule) > 25:
                    continue
                with self.subTest(rule=i, optimize=optimize):
                    dnf = postprocess(
                        rule, self.binarizer, optimize=optimize, canonical="dnf"
                    )
                    cnf = postprocess(
                        rule, self.binarizer, optimize=optimize, canonical="cnf"
                    )
                    self.assertTrue(is_normal_form(dnf, Or, And), str(dnf))
                    self.assertTrue(is_normal_form(cnf, And, Or), str(cnf))

    def test_equivalent_rules_have_the_same_canonical_form(self):
        rng = np.random.default_rng(1)
        for i, rule in enumerate(self.rules(150, depth=4)):
            other = rewrite(rule, rng)
            np.testing.assert_array_equal(
                predict(other, self.data), predict(rule, self.data)
            )
            for canonical in ("dnf", "cnf"):
                with self.subTest(rule=i, canonical=canonical):
                    self.assertEqual(
                        str(postprocess(other, self.binarizer, canonical=canonical)),
                        str(postprocess(rule, self.binarizer, canonical=canonical)),
                    )
                    self.assertEqual(
                        postprocess_to_str(other, self.binarizer, canonical=canonical),
                        postprocess_to_str(rule, self.binarizer, canonical=canonical),
                    )

    def test_canonical_terms_are_all_needed(self):
        # Removing a term (DNF) or a clause (CNF), or a literal from it, changes what
        # the rule computes.
        for i, rule in enumerate(self.rules(60, depth=4)):
            for canonical, outer in (("dnf", Or), ("cnf", And)):
                result = postprocess(rule, self.binarizer, canonical=canonical)
                if not isinstance(result, outer) or result.negated:
                    continue
                expected = predict(result, self.data)
                for j in range(len(result.subrules)):
                    with self.subTest(rule=i, canonical=canonical, part=j):
                        parts = [
                            p.copy() for k, p in enumerate(result.subrules) if k != j
                        ]
                        smaller = parts[0] if len(parts) == 1 else outer(parts)
                        self.assertFalse(
                            np.array_equal(predict(smaller, self.data), expected)
                        )

    def test_canonical_terms_are_prime(self):
        # Allowing one more cell of a column in a DNF term makes it true on a row where
        # the rule is false.
        cells = Cells(self.binarizer.get_feature_specs(), self.data)
        for i, rule in enumerate(self.rules(100, depth=4)):
            expected = predict(rule, self.data)
            if expected.all() or not expected.any():
                continue  # A constant is Or(0, ~0) or And(0, ~0), not prime terms.
            dnf = postprocess(rule, self.binarizer, canonical="dnf")
            terms = dnf.subrules if isinstance(dnf, Or) else [dnf]
            for j, term in enumerate(terms):
                allowed = cells.of_term(term)
                for column, cell in cells.widenings(allowed):
                    with self.subTest(rule=i, term=j, column=column, cell=cell):
                        wider = dict(allowed)
                        wider[column] = allowed[column] | {cell}
                        self.assertTrue((cells.rows_in(wider) & ~expected).any())

    def test_input_rule_is_not_modified(self):
        for rule in self.rules(30, depth=4):
            before = str(rule)
            for optimize, canonical in MODES:
                postprocess(
                    rule, self.binarizer, optimize=optimize, canonical=canonical
                )
                postprocess_to_str(
                    rule, self.binarizer, optimize=optimize, canonical=canonical
                )
            self.assertEqual(str(rule), before)


class TestIndependentFeatures(unittest.TestCase):
    """Without features, each literal is an independent boolean feature."""

    def test_same_predictions_on_every_assignment(self):
        rng = np.random.default_rng(2)
        num_features = 5
        data = np.array(list(itertools.product([False, True], repeat=num_features)))
        for i in range(150):
            rule = random_rule(rng, num_features, depth=4)
            expected = predict(rule, data)
            for optimize, canonical in MODES:
                if not optimize and canonical is not None and len(rule) > 25:
                    continue
                with self.subTest(rule=i, optimize=optimize, canonical=canonical):
                    result = postprocess(rule, optimize=optimize, canonical=canonical)
                    np.testing.assert_array_equal(predict(result, data), expected)

    def test_canonical_form_equals_that_of_the_truth_table(self):
        # The DNF of all true rows (and the CNF of all false rows) is as different from
        # a random rule as an equivalent rule can be.
        rng = np.random.default_rng(3)
        num_features = 5
        data = np.array(list(itertools.product([False, True], repeat=num_features)))
        for i in range(150):
            rule = random_rule(rng, num_features, depth=4)
            values = predict(rule, data)
            if values.all() or not values.any():
                continue
            minterms = Or(
                [
                    And([Literal(value=f, negated=not row[f]) for f in range(5)])
                    for row in data[values]
                ]
                + [And([Literal(value=0), Literal(value=0, negated=True)])]
            )
            maxterms = And(
                [
                    Or([Literal(value=f, negated=bool(row[f])) for f in range(5)])
                    for row in data[~values]
                ]
                + [Or([Literal(value=0), Literal(value=0, negated=True)])]
            )
            for canonical in ("dnf", "cnf"):
                with self.subTest(rule=i, canonical=canonical):
                    expected = str(postprocess(rule, canonical=canonical))
                    self.assertEqual(
                        str(postprocess(minterms, canonical=canonical)), expected
                    )
                    self.assertEqual(
                        str(postprocess(maxterms, canonical=canonical)), expected
                    )

    def test_literals_are_written_by_index(self):
        rule = And([Literal(value=0), Literal(value=1), Literal(value=0)])
        self.assertEqual(str(postprocess(rule)), "And(0, 1)")
        self.assertEqual(postprocess_to_str(rule), "And(0, 1)")
        self.assertEqual(postprocess_to_str(rule, optimize=False), "And(0, 1, 0)")

    def test_feature_names(self):
        rule = And([Literal(value=0), Literal(value=1, negated=True), Literal(value=0)])
        self.assertEqual(postprocess_to_str(rule, ["a", "b"]), "And(a, ~b)")


# ---------------------------------------------------------------------------
# Behaviors, one example each
# ---------------------------------------------------------------------------


def bins(column: str, *edges: float) -> list:
    edges = [-inf, *edges, inf]
    return [BinFeature(column, a, b) for a, b in itertools.pairwise(edges)]


class TestOptimize(unittest.TestCase):
    # Features 0-3: the bins of x (edges 1, 2, 3); 4-6: the categories of c; 7: flag.
    features = (
        *bins("x", 1.0, 2.0, 3.0),
        CategoryFeature("c", "r"),
        CategoryFeature("c", "g"),
        CategoryFeature("c", "b"),
        BooleanFeature("flag"),
    )
    names = tuple(str(f) for f in features)

    def check(self, rule: Rule, expected_rule: str, expected_str: str):
        self.assertEqual(
            postprocess(rule, self.features).to_str(self.names), expected_rule
        )
        self.assertEqual(postprocess_to_str(rule, self.features), expected_str)

    def test_repeated_literal(self):
        rule = And([Literal(value=0), Literal(value=7), Literal(value=0)])
        self.check(rule, "And(x < 1.0, flag)", "And(x < 1.0, flag)")

    def test_bins_exclude_each_other(self):
        rule = And([Literal(value=0), Literal(value=1), Literal(value=7)])
        self.check(rule, "And(x < 1.0, ~x < 1.0)", "False")

    def test_small_constant_rule_is_returned_as_is(self):
        rule = And([Literal(value=0), Literal(value=1)])
        self.check(rule, "And(x < 1.0, 1.0 <= x < 2.0)", "False")
        rule = Or([Literal(value=7), Literal(value=7, negated=True)])
        self.check(rule, "Or(flag, ~flag)", "True")

    def test_excluded_bins(self):
        rule = And([Literal(value=i, negated=True) for i in range(3)])
        self.check(rule, "3.0 <= x", "3.0 <= x")

    def test_all_bins(self):
        rule = And([Or([Literal(value=i) for i in range(4)]), Literal(value=7)])
        self.check(rule, "flag", "flag")

    def test_adjacent_bins_become_one_interval_in_the_string(self):
        features = bins("x", 1.0, 2.0, 3.0, 4.0)
        rule = Or([Literal(value=1), Literal(value=2), Literal(value=3)])
        result = postprocess(rule, features)
        self.assertEqual(
            result.to_str([str(f) for f in features]), "And(~x < 1.0, ~4.0 <= x)"
        )
        self.assertEqual(postprocess_to_str(rule, features), "1.0 <= x < 4.0")

    def test_bins_that_are_not_adjacent(self):
        rule = And([Or([Literal(value=0), Literal(value=2)]), Literal(value=7)])
        # In the rule, excluding two bins takes one node less than an Or of two bins.
        self.check(
            rule,
            "And(~1.0 <= x < 2.0, ~3.0 <= x, flag)",
            "And(Or(x < 1.0, 2.0 <= x < 3.0), flag)",
        )

    def test_categories(self):
        rule = And([Or([Literal(value=4), Literal(value=5)]), Literal(value=7)])
        # Two of three categories: the excluded one is shorter.
        self.check(rule, "And(~c=b, flag)", "And(~c=b, flag)")
        rule = Or([Literal(value=4), Literal(value=5), Literal(value=6)])
        self.check(rule, "Or(c=r, ~c=r)", "True")

    def test_category_set(self):
        features = [CategoryFeature("c", v) for v in "rgbk"]
        rule = Or([Literal(value=0), Literal(value=1)])
        self.assertEqual(postprocess_to_str(rule, features), "c in {r, g}")
        rule = Or([Literal(value=0), Literal(value=1), Literal(value=2)])
        self.assertEqual(postprocess_to_str(rule, features), "~c=k")

    def test_literal_decided_by_its_siblings(self):
        # Inside And(x < 1, ...), Or(x < 2, c=r) always holds.
        rule = And(
            [Literal(value=0), Or([Literal(value=1, negated=True), Literal(value=4)])]
        )
        self.check(rule, "x < 1.0", "x < 1.0")
        # Inside Or(flag, ...), And(~flag, c=r) is c=r.
        rule = Or(
            [Literal(value=7), And([Literal(value=7, negated=True), Literal(value=4)])]
        )
        self.check(rule, "Or(flag, c=r)", "Or(flag, c=r)")

    def test_tautology_inside_a_rule_is_removed(self):
        rule = And(
            [Literal(value=4), Or([Literal(value=7), Literal(value=7, negated=True)])]
        )
        self.check(rule, "c=r", "c=r")

    def test_absorption(self):
        a = And([Literal(value=4), Literal(value=7)])
        rule = Or([a, And([a.copy(), Literal(value=0)])])
        self.check(rule, "And(c=r, flag)", "And(c=r, flag)")
        rule = And(
            [
                Or([Literal(value=0), Literal(value=7)]),
                Or([Or([Literal(value=0), Literal(value=7)]), Literal(value=5)]),
            ]
        )
        self.check(rule, "Or(x < 1.0, flag)", "Or(x < 1.0, flag)")

    def test_operator_and_its_negation(self):
        a = Or([Literal(value=4), Literal(value=7)])
        rule = And(
            [
                a,
                Literal(value=0),
                And([Literal(value=4, negated=True), Literal(value=7, negated=True)]),
            ]
        )
        self.check(rule, "And(c=r, ~c=r)", "False")

    def test_negations_are_pushed_down(self):
        rule = And([Literal(value=4), Literal(value=7)], negated=True)
        self.check(rule, "Or(~c=r, ~flag)", "Or(~c=r, ~flag)")

    def test_constant_rule_uses_its_first_literal(self):
        rule = Or(
            [
                Literal(value=5),
                And([Literal(value=4), Literal(value=7)]),
                Literal(value=5, negated=True),
            ]
        )
        self.check(rule, "Or(c=g, ~c=g)", "True")


class TestMissingValues(unittest.TestCase):
    # Features 0: x is missing; 1-3: the bins of x (edges 1, 2).
    features = (MissingFeature("x"), *bins("x", 1.0, 2.0))
    names = tuple(str(f) for f in features)

    def test_negated_bin_includes_missing_values(self):
        rule = Literal(value=1, negated=True)
        self.assertEqual(postprocess_to_str(rule, self.features), "~x < 1.0")
        # All bins do not cover the missing values.
        rule = Or([Literal(value=1), Literal(value=2), Literal(value=3)])
        self.assertEqual(postprocess_to_str(rule, self.features), "x is not NA")
        self.assertEqual(
            postprocess(rule, self.features).to_str(self.names), "~x_is_NA"
        )

    def test_missing_value_written_explicitly(self):
        rule = Or([Literal(value=2), Literal(value=0)])
        self.assertEqual(
            postprocess_to_str(rule, self.features), "Or(1.0 <= x < 2.0, x is NA)"
        )

    def test_missing_and_present(self):
        rule = Or(
            [Literal(value=0), Literal(value=1), Literal(value=2), Literal(value=3)]
        )
        self.assertEqual(postprocess_to_str(rule, self.features), "True")

    def test_column_with_only_a_missing_indicator(self):
        features = [MissingFeature("id"), BooleanFeature("flag")]
        rule = And([Literal(value=0, negated=True), Literal(value=1)])
        self.assertEqual(postprocess_to_str(rule, features), "And(id is not NA, flag)")
        self.assertEqual(postprocess_to_str(Literal(value=0), features), "id is NA")


class TestCanonical(unittest.TestCase):
    features = (*bins("x", 1.0, 2.0), BooleanFeature("a"), BooleanFeature("b"))
    names = tuple(str(f) for f in features)

    def test_dnf_and_cnf(self):
        # And(Or(a, b), Or(a, x < 1))
        rule = And(
            [
                Or([Literal(value=3), Literal(value=4)]),
                Or([Literal(value=3), Literal(value=0)]),
            ]
        )
        dnf = postprocess(rule, self.features, canonical="dnf")
        cnf = postprocess(rule, self.features, canonical=CanonicalForm.CNF)
        self.assertEqual(dnf.to_str(self.names), "Or(a, And(x < 1.0, b))")
        self.assertEqual(cnf.to_str(self.names), "And(Or(x < 1.0, a), Or(a, b))")
        self.assertEqual(
            postprocess_to_str(rule, self.features, canonical="DNF"),
            "Or(a, And(x < 1.0, b))",
        )

    def test_canonical_form_does_not_depend_on_how_the_rule_is_written(self):
        # Or(And(a, b), And(~a, b)) and And(b, Or(a, ~a, x < 1)) are both b.
        first = Or(
            [
                And([Literal(value=3), Literal(value=4)]),
                And([Literal(value=3, negated=True), Literal(value=4)]),
            ]
        )
        second = And(
            [
                Literal(value=4),
                Or(
                    [Literal(value=3), Literal(value=3, negated=True), Literal(value=0)]
                ),
            ]
        )
        for canonical in ("dnf", "cnf"):
            self.assertEqual(
                str(postprocess(first, self.features, canonical=canonical)), "4"
            )
            self.assertEqual(
                str(postprocess(second, self.features, canonical=canonical)), "4"
            )

    def test_consensus_finds_terms_the_rule_does_not_spell_out(self):
        # Or(And(a, b), And(~a, x < 1)) also holds where b and x < 1 do.
        rule = Or(
            [
                And([Literal(value=3), Literal(value=4)]),
                And([Literal(value=3, negated=True), Literal(value=0)]),
            ]
        )
        dnf = postprocess(rule, self.features, canonical="dnf")
        self.assertEqual(dnf.to_str(self.names), "Or(And(a, b), And(x < 1.0, ~a))")

    def test_term_with_several_cells_keeps_the_dnf_shape(self):
        # x >= 1 is two bins: inside a term it is written with the excluded bin.
        rule = And([Or([Literal(value=1), Literal(value=2)]), Literal(value=3)])
        dnf = postprocess(rule, self.features, canonical="dnf")
        self.assertEqual(dnf.to_str(self.names), "And(~x < 1.0, a)")
        self.assertEqual(
            postprocess_to_str(rule, self.features, canonical="dnf"), "And(1.0 <= x, a)"
        )

    def test_constant_canonical_form(self):
        rule = Or([Literal(value=3), Literal(value=3, negated=True), Literal(value=4)])
        self.assertEqual(
            str(postprocess(rule, self.features, canonical="dnf")), "Or(0, ~0)"
        )
        self.assertEqual(
            str(postprocess(rule, self.features, canonical="cnf")), "Or(0, ~0)"
        )
        rule = And([Literal(value=0), Literal(value=1)])
        self.assertEqual(
            str(postprocess(rule, self.features, canonical="dnf")), "And(0, ~0)"
        )
        self.assertEqual(
            postprocess_to_str(rule, self.features, canonical="cnf"), "False"
        )

    def test_mechanical_conversion_removes_nothing(self):
        # ~And(a, Or(b, a)) is Or(~a, And(~b, ~a)).
        rule = And(
            [Literal(value=3), Or([Literal(value=4), Literal(value=3)])], negated=True
        )
        dnf = postprocess(rule, self.features, optimize=False, canonical="dnf")
        cnf = postprocess(rule, self.features, optimize=False, canonical="cnf")
        self.assertEqual(dnf.to_str(self.names), "Or(~a, And(~b, ~a))")
        self.assertEqual(cnf.to_str(self.names), "And(Or(~a, ~b), Or(~a, ~a))")
        self.assertEqual(
            postprocess_to_str(rule, self.features, optimize=False, canonical="cnf"),
            "And(Or(~a, ~b), Or(~a, ~a))",
        )

    def test_flags_can_be_positional(self):
        rule = And([Or([Literal(value=3), Literal(value=4)]), Literal(value=3)])
        self.assertEqual(
            str(postprocess(rule, self.features, True, "dnf")),
            str(postprocess(rule, self.features, optimize=True, canonical="dnf")),
        )
        self.assertEqual(
            postprocess_to_str(rule, self.features, False, "cnf", 0),
            postprocess_to_str(
                rule, self.features, optimize=False, canonical="cnf", indent=0
            ),
        )

    def test_without_optimize_and_canonical_the_rule_is_copied(self):
        rule = And([Literal(value=3), Literal(value=3)])
        result = postprocess(rule, self.features, optimize=False)
        self.assertIsNot(result, rule)
        self.assertEqual(str(result), "And(3, 3)")
        self.assertEqual(
            postprocess_to_str(rule, self.features, optimize=False), "And(a, a)"
        )


class TestIndentation(unittest.TestCase):
    def test_indent_matches_rule_to_str(self):
        rule = And(
            [Literal(value=0), Or([Literal(value=1), Literal(value=2, negated=True)])]
        )
        for indent in (-1, 0, 2):
            self.assertEqual(
                postprocess_to_str(rule, indent=indent), rule.to_str(indent=indent)
            )
            self.assertEqual(
                postprocess_to_str(rule, optimize=False, indent=indent),
                rule.to_str(indent=indent),
            )


class TestValidation(unittest.TestCase):
    rule = And([Literal(value=0), Literal(value=1)])

    def test_rule_type(self):
        with self.assertRaises(TypeError):
            postprocess("And(0, 1)")

        class Xor(Rule):
            pass

        with self.assertRaises(TypeError):
            postprocess(Xor([Literal(value=0), Literal(value=1)]))
        with self.assertRaises(TypeError):
            postprocess_to_str(
                And([Literal(value=0), Xor([Literal(value=1), Literal(value=0)])]),
                optimize=False,
            )

    def test_flags(self):
        with self.assertRaises(TypeError):
            postprocess(self.rule, optimize=1)
        with self.assertRaises(ValueError):
            postprocess(self.rule, canonical="anf")
        with self.assertRaises(TypeError):
            postprocess(self.rule, canonical=True)
        with self.assertRaises(ValueError):
            postprocess_to_str(self.rule, indent=-2)
        with self.assertRaises(TypeError):
            postprocess_to_str(self.rule, indent=1.0)

    def test_features(self):
        with self.assertRaises(ValueError):
            postprocess(self.rule, ["only one"])
        with self.assertRaises(TypeError):
            postprocess(self.rule, "ab")
        with self.assertRaises(TypeError):
            postprocess(self.rule, [1, 2])
        with self.assertRaises(NotFittedError):
            postprocess(self.rule, StandardBinarizer())

    def test_inconsistent_features(self):
        with self.assertRaises(ValueError):
            postprocess(
                self.rule, [BinFeature("x", -inf, inf), CategoryFeature("x", "a")]
            )
        with self.assertRaises(ValueError):
            postprocess(
                self.rule, [BinFeature("x", -inf, 1.0), BinFeature("x", 2.0, inf)]
            )


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


class TestBreastCancer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        y = y.to_numpy() == 1
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    def test_rule_from_the_docs(self):
        # The HGP rule of docs/experiments/breast-cancer.md, on the same bins.
        binarizer = StandardBinarizer(progress_bar=False)
        binarizer.fit_transform(self.X_train, self.y_train)
        names = binarizer.get_feature_names_out()

        def lit(name, negated=False):
            return Literal(value=names.index(name), negated=negated)

        rule = And(
            [
                lit("115.35 <= worst perimeter", True),
                And(
                    [
                        lit("0.125 <= mean compactness < 0.1293", True),
                        lit("1.807 <= texture error < 1.9", True),
                        Or(
                            [
                                Or(
                                    [
                                        Or(
                                            [
                                                lit("16.8 <= worst radius < 17.51"),
                                                lit("696.25 <= mean area"),
                                            ],
                                            negated=True,
                                        ),
                                        lit("mean texture < 16.395"),
                                    ]
                                ),
                                lit("29.3 <= worst texture < 30.91"),
                            ]
                        ),
                        lit("739.2 <= worst area < 743.45", True),
                        lit("0.176 <= worst concave points", True),
                    ]
                ),
                lit("17.51 <= worst radius", True),
                lit("17.51 <= worst radius", True),
                lit("1.807 <= texture error < 1.9", True),
                lit("739.2 <= worst area < 743.45", True),
                Or(
                    [
                        lit("mean texture < 16.395", True),
                        lit("1.807 <= texture error < 1.9", True),
                    ]
                ),
            ]
        )
        self.assertEqual((len(rule), literal_count(rule)), (21, 15))
        result = postprocess(rule, binarizer)
        self.assertEqual((len(result), literal_count(result)), (13, 10))
        # The string shown in docs/guide/postprocessing.md.
        self.assertEqual(
            postprocess_to_str(rule, binarizer),
            "And(worst perimeter < 115.35, ~0.125 <= mean compactness < 0.1293, "
            "~1.807 <= texture error < 1.9, Or(And(worst radius < 16.8, "
            "mean area < 696.25), mean texture < 16.395, 29.3 <= worst texture < 30.91), "
            "~739.2 <= worst area < 743.45, worst concave points < 0.176, "
            "worst radius < 17.51)",
        )
        data = binarizer.transform(pd.concat([self.X_train, self.X_test])).to_numpy()
        np.testing.assert_array_equal(predict(result, data), predict(rule, data))

    def test_classifier(self):
        from hgp_lib import BooleanRuleClassifier
        from hgp_lib.configs import TrainerConfig

        clf = BooleanRuleClassifier(
            TrainerConfig(num_epochs=10, progress_bar=False),
            StandardBinarizer(progress_bar=False),
        )
        clf.fit(self.X_train, self.y_train)
        data = clf.binarizer.transform(self.X_test).to_numpy()
        expected = clf.predict(self.X_test)
        for optimize, canonical in [
            (True, None),
            (True, "dnf"),
            (True, "cnf"),
            (False, None),
        ]:
            with self.subTest(optimize=optimize, canonical=canonical):
                result = clf.postprocess(optimize=optimize, canonical=canonical)
                np.testing.assert_array_equal(predict(result, data), expected)
                self.assertEqual(
                    clf.postprocess_to_str(optimize=optimize, canonical=canonical),
                    postprocess_to_str(
                        clf.rule, clf.binarizer, optimize=optimize, canonical=canonical
                    ),
                )
        self.assertLessEqual(len(clf.postprocess()), len(clf.rule))
        self.assertEqual(clf.postprocess(optimize=False).to_str(), clf.rule.to_str())


if __name__ == "__main__":
    unittest.main()
