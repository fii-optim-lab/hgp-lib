from itertools import pairwise, product
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp
from sympy import sympify
from sympy.core.relational import Rel
from sympy.logic.boolalg import is_cnf, is_dnf

from hgp_lib.postprocessing import ThresholdSimplifier, simplify_threshold_formula
from hgp_lib.postprocessing.rule_to_sympy import (
    _convert_rule,
    _feature_base_name,
    _number,
    _safe_symbol_name,
    feature_to_sympy,
    rule_to_sympy,
)


def _collect_bounds(expr, bounds):
    for atom in expr.atoms(Rel):
        lhs, rhs = atom.lhs, atom.rhs

        if lhs.is_Symbol and rhs.is_number:
            bounds.setdefault(lhs, set()).add(rhs)

        elif rhs.is_Symbol and lhs.is_number:
            bounds.setdefault(rhs, set()).add(lhs)

        else:
            raise ValueError(f"Expected variable-vs-number comparison, got: {atom}")


def _sample_points(bounds):
    points = {}

    for var, values in bounds.items():
        values = sorted(values, key=float)

        samples = set(values)

        # one point below the smallest boundary
        samples.add(values[0] - 1)

        # one point between every pair of boundaries
        for a, b in pairwise(values):
            samples.add((a + b) / 2)

        # one point above the largest boundary
        samples.add(values[-1] + 1)

        points[var] = sorted(samples, key=float)

    return points


def equivalent(e1, e2):
    """
    Exact equivalence test for Boolean combinations of comparisons
    between real variables and numeric constants.

    Supported:

        x < c
        x <= c
        x > c
        x >= c

    combined with &, |, and parentheses.
    """
    e1 = sympify(e1)
    e2 = sympify(e2)

    bounds = {}

    _collect_bounds(e1, bounds)
    _collect_bounds(e2, bounds)

    points = _sample_points(bounds)
    variables = list(points)

    # no variables/bounds
    if not variables:
        return bool(e1 == e2)

    # evaluate both expressions at every relevant combination
    # of one-dimensional sample points
    for values in product(*(points[v] for v in variables)):
        substitution = dict(zip(variables, values))

        v1 = bool(e1.subs(substitution))
        v2 = bool(e2.subs(substitution))

        if v1 != v2:
            return False

    return True


# ====

# class TestPostprecessing(unittest.TestCase):
#     def setUp(self):
#         # Seed random generators for reproducibility
#         random.seed(42)
#         np.random.seed(42)


@pytest.fixture
def ts():
    return ThresholdSimplifier()


@pytest.fixture
def x():
    return sp.Symbol("x", real=True)


@pytest.fixture
def xy():
    return sp.symbols("x y", real=True)


def atoms_for(x):
    return {
        "lt": x < 3,
        "le": x <= 3,
        "gt": x > 3,
        "ge": x >= 3,
    }


def eval_bool(expr, x_value, y_value=None):
    expr = sp.sympify(expr)

    symbols = {str(s): s for s in expr.free_symbols}

    subs = []

    if x_value is not None and "x" in symbols:
        subs.append((symbols["x"], x_value))

    if y_value is not None and "y" in symbols:
        subs.append((symbols["y"], y_value))

    return bool(expr.subs(subs))


def assert_equivalent_on_grid(actual, expected, values=(-5, -1, 0, 1, 2, 3, 4, 5, 10)):
    for xv in values:
        assert eval_bool(actual, xv) == eval_bool(expected, xv), (
            f"Mismatch at x={xv}: actual={actual}, expected={expected}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (sp.true, sp.true),
        (sp.false, sp.false),
    ],
)
def test_simplify_boolean_constants(ts, expr, expected):
    result = ts.simplify(expr)
    assert result["best_cnf"] == expected
    assert result["best_dnf"] == expected


def test_simplify_single_atom(ts, x):
    expr = x >= 3
    result = ts.simplify(expr)
    assert result["best_cnf"] == expr
    assert result["best_dnf"] == expr


def test_simplify_negated_atom(ts, x):
    expr = sp.Not(x < 3)
    result = ts.simplify(expr)
    assert_equivalent_on_grid(result["best_dnf"], x >= 3)
    assert_equivalent_on_grid(result["best_cnf"], x >= 3)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            sp.And(sp.true, sp.Symbol("x", real=True) >= 0),
            sp.Symbol("x", real=True) >= 0,
        ),
        (
            sp.Or(sp.false, sp.Symbol("x", real=True) >= 0),
            sp.Symbol("x", real=True) >= 0,
        ),
        (sp.And(sp.false, sp.Symbol("x", real=True) >= 0), sp.false),
        (sp.Or(sp.true, sp.Symbol("x", real=True) >= 0), sp.true),
    ],
)
def test_simplify_boolean_identity_cases(expr, expected):
    result = simplify_threshold_formula(expr)
    assert_equivalent_on_grid(result["best_dnf"], expected)
    assert_equivalent_on_grid(result["best_cnf"], expected)


def test_simplify_returns_both_normal_forms(ts, x):
    expr = sp.Or(sp.And(x > 1, x < 5), x >= 10)
    result = ts.simplify(expr)

    assert set(result) == {"best_cnf", "best_dnf"}
    assert result["best_cnf"] is not None
    assert result["best_dnf"] is not None
    assert_equivalent_on_grid(result["best_cnf"], expr, values=range(-2, 13))
    assert_equivalent_on_grid(result["best_dnf"], expr, values=range(-2, 13))


def test_simplify_normalizes_non_nnf_input(ts, x):
    expr = sp.Not(sp.And(x < 1, x > -1))
    result = ts.simplify(expr)

    for value in (-2, -1, 0, 1, 2):
        expected = bool(expr.subs(x, value))
        assert bool(result["best_cnf"].subs(x, value)) == expected
        assert bool(result["best_dnf"].subs(x, value)) == expected


def test_max_iterations_zero_still_returns_result_structure(x):
    result = ThresholdSimplifier(max_iterations=0).simplify(x > 0)
    assert set(result) == {"best_cnf", "best_dnf"}


def test_repeated_simplify_calls_do_not_use_stale_expression_state(x):
    simplifier = ThresholdSimplifier()
    first = simplifier.simplify(x > 0)
    second = simplifier.simplify(x < 0)

    assert_equivalent_on_grid(second["best_dnf"], x < 0)
    assert_equivalent_on_grid(second["best_cnf"], x < 0)
    assert (
        first["best_dnf"] is not second["best_dnf"]
        or first["best_dnf"] != second["best_dnf"]
    )


# ---------------------------------------------------------------------------
# Atom handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        sp.Symbol("x", real=True) < 1,
        sp.Symbol("x", real=True) <= 1,
        sp.Symbol("x", real=True) > 1,
        sp.Symbol("x", real=True) >= 1,
    ],
)
def test_is_atom_accepts_all_supported_relations(expr):
    assert ThresholdSimplifier._is_atom(expr) is True


@pytest.mark.parametrize(
    "expr",
    [
        sp.Symbol("x", real=True),
        sp.Symbol("x", real=True) + 1,
        sp.true,
        sp.false,
        sp.Symbol("x", real=True) == 1,
    ],
)
def test_is_atom_rejects_non_threshold_expressions(expr):
    assert ThresholdSimplifier._is_atom(expr) is False


def test_parts_rejects_non_atom(x):
    with pytest.raises(TypeError, match="Not a threshold atom"):
        ThresholdSimplifier._parts(x + 1)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (sp.Symbol("x", real=True) < 3, sp.Symbol("x", real=True) >= 3),
        (sp.Symbol("x", real=True) <= 3, sp.Symbol("x", real=True) > 3),
        (sp.Symbol("x", real=True) > 3, sp.Symbol("x", real=True) <= 3),
        (sp.Symbol("x", real=True) >= 3, sp.Symbol("x", real=True) < 3),
    ],
)
def test_negate_atom(expr, expected):
    assert ThresholdSimplifier._negate_atom(expr) == expected


def test_negate_atom_has_expected_double_negation_property(x):
    for atom in atoms_for(x).values():
        assert (
            ThresholdSimplifier._negate_atom(ThresholdSimplifier._negate_atom(atom))
            == atom
        )


# ---------------------------------------------------------------------------
# Exact threshold comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (sp.Integer(1), sp.Integer(1), 0),
        (sp.Integer(1), sp.Integer(2), -1),
        (sp.Integer(2), sp.Integer(1), 1),
        (sp.Rational(1, 2), sp.Integer(1), -1),
        (sp.Integer(1), sp.Rational(1, 2), 1),
        (sp.Integer(2), sp.sqrt(3), 1),
        (sp.sqrt(3), sp.Integer(2), -1),
    ],
)
def test_cmp(a, b, expected):
    assert ThresholdSimplifier._cmp(a, b) == expected


def test_cmp_raises_for_incomparable_symbolic_thresholds():
    a, b = sp.symbols("a b", real=True)
    with pytest.raises(ValueError, match="Cannot compare thresholds"):
        ThresholdSimplifier._cmp(a, b)


# ---------------------------------------------------------------------------
# Interval construction and mutation
# ---------------------------------------------------------------------------


def test_empty_interval():
    assert ThresholdSimplifier._empty_interval() == [None, False, None, False]


@pytest.mark.parametrize(
    ("initial", "value", "closed", "expected"),
    [
        ([None, False, None, False], 3, True, [3, True, None, False]),
        ([5, True, None, False], 3, True, [5, True, None, False]),
        ([5, True, None, False], 7, True, [7, True, None, False]),
        ([5, True, None, False], 5, False, [5, False, None, False]),
        ([5, False, None, False], 5, True, [5, False, None, False]),
    ],
)
def test_set_lower(ts, initial, value, closed, expected):
    ts._set_lower(initial, sp.Integer(value), closed)
    assert initial == expected


@pytest.mark.parametrize(
    ("initial", "value", "closed", "expected"),
    [
        ([None, False, None, False], 3, True, [None, False, 3, True]),
        ([None, False, 5, True], 7, True, [None, False, 5, True]),
        ([None, False, 5, True], 3, True, [None, False, 3, True]),
        ([None, False, 5, True], 5, False, [None, False, 5, False]),
        ([None, False, 5, False], 5, True, [None, False, 5, False]),
    ],
)
def test_set_upper(ts, initial, value, closed, expected):
    ts._set_upper(initial, sp.Integer(value), closed)
    assert initial == expected


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ([None, False, None, False], True),
        ([1, True, 2, True], True),
        ([1, False, 1, False], False),
        ([1, True, 1, False], False),
        ([1, False, 1, True], False),
        ([1, True, 1, True], True),
        ([2, True, 1, True], False),
    ],
)
def test_valid_interval(ts, interval, expected):
    assert ts._valid_interval(interval) is expected


@pytest.mark.parametrize(
    ("atoms", "expected"),
    [
        (
            [sp.Symbol("x", real=True) > 1, sp.Symbol("x", real=True) < 5],
            {sp.Symbol("x", real=True): [sp.Integer(1), False, sp.Integer(5), False]},
        ),
        (
            [sp.Symbol("x", real=True) >= 1, sp.Symbol("x", real=True) <= 5],
            {sp.Symbol("x", real=True): [sp.Integer(1), True, sp.Integer(5), True]},
        ),
        (
            [sp.Symbol("x", real=True) > 1, sp.Symbol("x", real=True) >= 5],
            {sp.Symbol("x", real=True): [sp.Integer(5), True, None, False]},
        ),
        (
            [sp.Symbol("x", real=True) < 5, sp.Symbol("x", real=True) <= 3],
            {sp.Symbol("x", real=True): [None, False, sp.Integer(3), True]},
        ),
    ],
)
def test_interval_from_atoms(ts, atoms, expected):
    assert ts._interval_from_atoms(frozenset(atoms)) == expected


@pytest.mark.parametrize(
    "atoms",
    [
        [sp.Symbol("x", real=True) > 5, sp.Symbol("x", real=True) < 3],
        [sp.Symbol("x", real=True) >= 5, sp.Symbol("x", real=True) <= 3],
        [sp.Symbol("x", real=True) > 3, sp.Symbol("x", real=True) <= 3],
        [sp.Symbol("x", real=True) >= 3, sp.Symbol("x", real=True) < 3],
    ],
)
def test_interval_from_atoms_detects_contradictions(ts, atoms):
    assert ts._interval_from_atoms(frozenset(atoms)) is None


def test_interval_from_atoms_handles_multiple_variables(ts):
    x, y = sp.symbols("x y", real=True)
    result = ts._interval_from_atoms(frozenset([x > 1, x <= 4, y >= -2, y < 10]))
    assert result == {
        x: [sp.Integer(1), False, sp.Integer(4), True],
        y: [sp.Integer(-2), True, sp.Integer(10), False],
    }


# ---------------------------------------------------------------------------
# Interval implication and term implication
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("interval", "atom", "expected"),
    [
        ([5, True, 10, True], sp.Symbol("x", real=True) >= 4, True),
        ([5, True, 10, True], sp.Symbol("x", real=True) > 4, True),
        ([5, True, 10, True], sp.Symbol("x", real=True) >= 5, True),
        ([5, True, 10, True], sp.Symbol("x", real=True) > 5, False),
        ([5, False, 10, True], sp.Symbol("x", real=True) > 5, True),
        ([5, False, 10, True], sp.Symbol("x", real=True) >= 5, True),
        ([5, True, 10, True], sp.Symbol("x", real=True) < 10, False),
        ([5, True, 10, False], sp.Symbol("x", real=True) < 10, True),
        ([5, True, 10, True], sp.Symbol("x", real=True) <= 10, True),
        ([None, False, 10, True], sp.Symbol("x", real=True) <= 10, True),
        ([None, False, 10, False], sp.Symbol("x", real=True) < 10, True),
        ([5, True, None, False], sp.Symbol("x", real=True) > 4, True),
        ([5, False, None, False], sp.Symbol("x", real=True) > 5, True),
    ],
)
def test_interval_implies_atom(ts, interval, atom, expected):
    assert ts._interval_implies_atom(interval, atom) is expected


def test_term_implies_term(ts):
    x, y = sp.symbols("x y", real=True)

    assert ts._term_implies_term(frozenset([x > 5]), frozenset([x > 3]))
    assert ts._term_implies_term(frozenset([x >= 5]), frozenset([x >= 5]))
    assert not ts._term_implies_term(frozenset([x > 3]), frozenset([x > 5]))
    assert not ts._term_implies_term(frozenset([x > 3]), frozenset([y > 3]))


def test_contradictory_source_term_implies_everything(ts, x):
    source = frozenset([x > 5, x < 3])
    target = frozenset([x > 100, x < -100])
    assert ts._term_implies_term(source, target) is True


# ---------------------------------------------------------------------------
# DNF term simplification and absorption
# ---------------------------------------------------------------------------


def test_simplify_term_keeps_only_strongest_bounds(ts, x):
    term = frozenset([x >= 3, x >= 5, x < 10, x <= 8])
    result = ts._simplify_term(term)

    assert result == frozenset([x >= 5, x <= 8])


def test_simplify_term_preserves_singleton_if_both_closed(ts, x):
    result = ts._simplify_term(frozenset([x >= 3, x <= 3]))
    assert result == frozenset([x >= 3, x <= 3])


def test_simplify_term_detects_open_singleton_contradiction(ts, x):
    assert ts._simplify_term(frozenset([x > 3, x <= 3])) is None
    assert ts._simplify_term(frozenset([x >= 3, x < 3])) is None


def test_simplify_term_drops_redundant_bounds_with_equal_threshold(ts, x):
    result = ts._simplify_term(frozenset([x > 3, x >= 3, x < 10, x <= 10]))
    assert result == frozenset([x > 3, x < 10])


def test_absorb_dnf_by_subset(ts, x):
    a = frozenset([x > 0])
    b = frozenset([x > 0, x < 10])
    assert ts._absorb_dnf(frozenset([a, b])) == frozenset([a])


def test_absorb_dnf_by_threshold_implication(ts, x):
    weaker = frozenset([x >= 0])
    stronger = frozenset([x >= 5])
    assert ts._absorb_dnf(frozenset([weaker, stronger])) == frozenset([weaker])


def test_absorb_dnf_is_order_independent(ts, x):
    a = frozenset([x >= 0])
    b = frozenset([x >= 0, x < 10])
    first = ts._absorb_dnf(frozenset([a, b]))
    second = ts._absorb_dnf(frozenset([b, a]))
    assert first == second


@pytest.mark.parametrize(
    ("dnf", "expected"),
    [
        (ThresholdSimplifier.DNF_FALSE, ThresholdSimplifier.DNF_FALSE),
        (ThresholdSimplifier.DNF_TRUE, ThresholdSimplifier.DNF_TRUE),
    ],
)
def test_simplify_dnf_constants(ts, dnf, expected):
    assert ts._simplify_dnf(dnf) == expected


def test_simplify_dnf_removes_contradictory_terms(ts, x):
    dnf = frozenset(
        [
            frozenset([x > 5, x < 3]),
            frozenset([x >= 0]),
        ]
    )
    assert ts._simplify_dnf(dnf) == frozenset([frozenset([x >= 0])])


def test_simplify_dnf_empty_term_means_true(ts, x):
    dnf = frozenset([frozenset(), frozenset([x > 0])])
    assert ts._simplify_dnf(dnf) == ThresholdSimplifier.DNF_TRUE


# ---------------------------------------------------------------------------
# CNF tautology detection and clause simplification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("atoms", "expected"),
    [
        ([sp.Symbol("x", real=True) < 3, sp.Symbol("x", real=True) >= 3], True),
        ([sp.Symbol("x", real=True) <= 3, sp.Symbol("x", real=True) > 3], True),
        ([sp.Symbol("x", real=True) < 3, sp.Symbol("x", real=True) >= 2], True),
        ([sp.Symbol("x", real=True) > 4, sp.Symbol("x", real=True) <= 3], False),
        ([sp.Symbol("x", real=True) >= 3], False),
        ([], False),
    ],
)
def test_union_covers_reals(ts, atoms, expected):
    assert ts._union_covers_reals(atoms) is expected


def test_union_covers_reals_handles_touching_closed_endpoint(ts, x):
    assert ts._union_covers_reals([x < 3, x >= 3]) is True


def test_union_covers_reals_handles_open_gap(ts, x):
    assert ts._union_covers_reals([x < 3, x > 3]) is False


def test_union_covers_reals_can_merge_multiple_intervals(ts, x):
    assert ts._union_covers_reals([x < 0, x >= 0, x > 10]) is True


def test_union_covers_reals_requires_negative_infinity_start(ts, x):
    assert ts._union_covers_reals([x >= 0]) is False


def test_clause_is_tautology_can_use_one_variable_among_many(ts):
    x, y = sp.symbols("x y", real=True)
    assert ts._clause_is_tautology(frozenset([x < 0, x >= 0, y > 100]))
    assert not ts._clause_is_tautology(frozenset([x < 0, y >= 0]))


def test_simplify_clause_removes_weaker_disjunct(ts, x):
    clause = frozenset([x >= 5, x >= 3])
    assert ts._simplify_clause(clause) == frozenset([x >= 3])


def test_simplify_clause_removes_redundant_strict_disjunct(ts, x):
    clause = frozenset([x > 5, x >= 5])
    assert ts._simplify_clause(clause) == frozenset([x >= 5])


def test_simplify_clause_returns_none_for_tautology(ts, x):
    assert ts._simplify_clause(frozenset([x < 3, x >= 3])) is None


def test_simplify_clause_handles_single_atom(ts, x):
    atom = x >= 3
    assert ts._simplify_clause(frozenset([atom])) == frozenset([atom])


def test_clause_implies_clause(ts, x):
    assert ts._clause_implies_clause(
        frozenset([x >= 5]),
        frozenset([x >= 3]),
    )
    assert not ts._clause_implies_clause(
        frozenset([x >= 3]),
        frozenset([x >= 5]),
    )


def test_empty_target_clause_is_not_implied(ts, x):
    assert ts._clause_implies_clause(frozenset([x >= 3]), frozenset()) is False


def test_absorb_cnf_by_implication(ts, x):
    strong = frozenset([x >= 5])
    weak = frozenset([x >= 3])
    assert ts._absorb_cnf(frozenset([strong, weak])) == frozenset([strong])


@pytest.mark.parametrize(
    ("cnf", "expected"),
    [
        (ThresholdSimplifier.CNF_TRUE, ThresholdSimplifier.CNF_TRUE),
        (ThresholdSimplifier.CNF_FALSE, ThresholdSimplifier.CNF_FALSE),
    ],
)
def test_simplify_cnf_constants(ts, cnf, expected):
    assert ts._simplify_cnf(cnf) == expected


def test_simplify_cnf_removes_tautological_clause(ts, x):
    cnf = frozenset(
        [
            frozenset([x < 0, x >= 0]),
            frozenset([x >= 5]),
        ]
    )
    assert ts._simplify_cnf(cnf) == frozenset([frozenset([x >= 5])])


def test_simplify_cnf_empty_clause_is_false(ts, x):
    cnf = frozenset([frozenset()])
    assert ts._simplify_cnf(cnf) == ThresholdSimplifier.CNF_FALSE


def test_simplify_cnf_all_tautologies_becomes_true(ts, x):
    cnf = frozenset(
        [
            frozenset([x < 0, x >= 0]),
            frozenset([x <= 10, x > 10]),
        ]
    )
    assert ts._simplify_cnf(cnf) == ThresholdSimplifier.CNF_TRUE


# ---------------------------------------------------------------------------
# DNF construction
# ---------------------------------------------------------------------------


def test_to_dnf_constants_atoms_and_cache(ts, x):
    assert ts._to_dnf(sp.true) == ts.DNF_TRUE
    assert ts._to_dnf(sp.false) == ts.DNF_FALSE

    atom = x >= 0
    expected = frozenset([frozenset([atom])])
    assert ts._to_dnf(atom) == expected
    assert atom in ts._dnf_cache
    assert ts._to_dnf(atom) is ts._dnf_cache[atom]


def test_to_dnf_or_combines_terms(ts, x):
    expr = sp.Or(x < 0, x >= 5)
    expected = frozenset(
        [
            frozenset([x < 0]),
            frozenset([x >= 5]),
        ]
    )
    assert ts._to_dnf(expr) == expected


def test_to_dnf_and_distributes_and_simplifies(ts, x):
    expr = sp.And(sp.Or(x < 0, x > 5), x <= 10)
    result = ts._to_dnf(expr)

    expected = frozenset(
        [
            frozenset([x < 0]),
            frozenset([x > 5, x <= 10]),
        ]
    )
    assert result == expected


def test_to_dnf_contradictory_and_becomes_false(ts, x):
    assert ts._to_dnf(sp.And(x > 5, x < 3)) == ts.DNF_FALSE


def test_to_dnf_true_and_x_is_x(ts, x):
    assert ts._to_dnf(sp.And(sp.true, x >= 0)) == frozenset([frozenset([x >= 0])])


def test_to_dnf_false_or_x_is_x(ts, x):
    assert ts._to_dnf(sp.Or(sp.false, x >= 0)) == frozenset([frozenset([x >= 0])])


def test_to_dnf_unsupported_expression_raises(ts, x):
    with pytest.raises(TypeError, match="Unsupported Boolean expression"):
        # circumvent automatic not evaluation
        # ts._to_dnf(sp.Not(x > 0))
        ts._to_dnf(sp.Basic.__new__(sp.Not, x > 0))


def test_dnf_and_truth_table_identities(ts, x):
    atom = frozenset([x > 0])
    assert ts._dnf_and(ts.DNF_FALSE, atom) == ts.DNF_FALSE
    assert ts._dnf_and(atom, ts.DNF_FALSE) == ts.DNF_FALSE
    assert ts._dnf_and(ts.DNF_TRUE, atom) == atom
    assert ts._dnf_and(atom, ts.DNF_TRUE) == atom


# ---------------------------------------------------------------------------
# CNF construction
# ---------------------------------------------------------------------------


def test_to_cnf_constants_atoms_and_cache(ts, x):
    assert ts._to_cnf(sp.true) == ts.CNF_TRUE
    assert ts._to_cnf(sp.false) == ts.CNF_FALSE

    atom = x >= 0
    expected = frozenset([frozenset([atom])])
    assert ts._to_cnf(atom) == expected
    assert atom in ts._cnf_cache
    assert ts._to_cnf(atom) is ts._cnf_cache[atom]


def test_to_cnf_and_combines_clauses(ts, x):
    expr = sp.And(x > 0, x < 10)
    expected = frozenset(
        [
            frozenset([x > 0]),
            frozenset([x < 10]),
        ]
    )
    assert ts._to_cnf(expr) == expected


def test_to_cnf_or_distributes_from_false_identity(ts, x):
    expr = sp.Or(sp.And(x > 0, x < 10), x >= 20)
    result = ts._to_cnf(expr)

    expected = frozenset(
        [
            frozenset([x > 0]),
            frozenset([x < 10, x >= 20]),
        ]
    )
    assert result == expected


# NOTE: happens in dnf step
# def test_to_cnf_contradictory_and_becomes_false(ts, x):
#     assert ts._to_cnf(sp.And(x > 5, x < 3)) == ts.CNF_FALSE


def test_to_cnf_true_or_x_is_true(ts, x):
    assert ts._to_cnf(sp.Or(sp.true, x >= 0)) == ts.CNF_TRUE


def test_to_cnf_false_and_x_is_false(ts, x):
    assert ts._to_cnf(sp.And(sp.false, x >= 0)) == ts.CNF_FALSE


def test_to_cnf_unsupported_expression_raises(ts, x):
    with pytest.raises(TypeError, match="Unsupported Boolean expression"):
        # circumvent automatic not evaluation
        # ts._to_cnf(sp.Not(x > 0))
        ts._to_cnf(sp.Basic.__new__(sp.Not, x > 0))


def test_cnf_or_truth_table_identities(ts, x):
    atom = frozenset([frozenset([x > 0])])
    assert ts._cnf_or(ts.CNF_TRUE, atom) == ts.CNF_TRUE
    assert ts._cnf_or(atom, ts.CNF_TRUE) == ts.CNF_TRUE
    assert ts._cnf_or(ts.CNF_FALSE, atom) == atom
    assert ts._cnf_or(atom, ts.CNF_FALSE) == atom


# ---------------------------------------------------------------------------
# Conversion back to SymPy
# ---------------------------------------------------------------------------


def test_and_expr_empty_is_true():
    assert ThresholdSimplifier._and_expr([]) == sp.true


def test_and_expr_sorts_atoms_deterministically():
    x = sp.Symbol("x", real=True)
    atoms = [x < 5, x > 1]
    assert ThresholdSimplifier._and_expr(atoms) == sp.And(*sorted(atoms, key=str))


def test_or_expr_empty_is_false():
    assert ThresholdSimplifier._or_expr([]) == sp.false


def test_or_expr_sorts_atoms_deterministically():
    x = sp.Symbol("x", real=True)
    atoms = [x < 5, x > 1]
    assert ThresholdSimplifier._or_expr(atoms) == sp.Or(*sorted(atoms, key=str))


def test_dnf_to_expr_constants(ts):
    assert ts._dnf_to_expr(ts.DNF_FALSE) == sp.false
    assert ts._dnf_to_expr(ts.DNF_TRUE) == sp.true


def test_dnf_to_expr_round_trip(ts, x):
    dnf = frozenset(
        [
            frozenset([x < 0]),
            frozenset([x >= 5, x <= 10]),
        ]
    )
    expr = ts._dnf_to_expr(dnf)
    assert ts._to_dnf(expr) == dnf


def test_cnf_to_expr_constants(ts):
    assert ts._cnf_to_expr(ts.CNF_TRUE) == sp.true
    assert ts._cnf_to_expr(ts.CNF_FALSE) == sp.false


def test_cnf_to_expr_round_trip(ts, x):
    cnf = frozenset(
        [
            frozenset([x < 0, x >= 5]),
            frozenset([x <= 10]),
        ]
    )
    expr = ts._cnf_to_expr(cnf)
    assert ts._to_cnf(expr) == cnf


def test_expr_key_is_structural_and_uses_strings(ts, x):
    dnf = frozenset([frozenset([x >= 1, x < 5])])
    key = ts._expr_key(dnf)
    assert key == frozenset([frozenset({str(x >= 1), str(x < 5)})])


# ---------------------------------------------------------------------------
# Integration / semantic properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        sp.Or(
            sp.And(sp.Symbol("x", real=True) > 1, sp.Symbol("x", real=True) < 5),
            sp.Symbol("x", real=True) >= 10,
        ),
        sp.And(
            sp.Or(sp.Symbol("x", real=True) < 0, sp.Symbol("x", real=True) >= 5),
            sp.Symbol("x", real=True) <= 20,
        ),
        sp.Not(sp.Or(sp.Symbol("x", real=True) <= 0, sp.Symbol("x", real=True) >= 10)),
        sp.Or(
            sp.And(sp.Symbol("x", real=True) >= 0, sp.Symbol("x", real=True) <= 3),
            sp.And(sp.Symbol("x", real=True) > 3, sp.Symbol("x", real=True) < 5),
        ),
        sp.And(
            sp.Or(sp.Symbol("x", real=True) < -2, sp.Symbol("x", real=True) >= 2),
            sp.Or(sp.Symbol("x", real=True) <= 5, sp.Symbol("x", real=True) > 8),
        ),
    ],
)
def test_public_simplification_preserves_semantics(expr):
    result = simplify_threshold_formula(expr)

    for value in range(-12, 13):
        expected = bool(expr.subs({"x": value}))
        assert bool(result["best_cnf"].subs({"x": value})) == expected
        assert bool(result["best_dnf"].subs({"x": value})) == expected


def test_two_variable_formula_preserves_semantics():
    x, y = sp.symbols("x y", real=True)
    expr = sp.Or(
        sp.And(x >= 0, x < 5),
        sp.And(y > 2, y <= 7),
    )

    result = simplify_threshold_formula(expr)

    for xv in range(-2, 9):
        for yv in range(-2, 10):
            expected = bool(expr.subs({x: xv, y: yv}))
            assert bool(result["best_cnf"].subs({x: xv, y: yv})) == expected
            assert bool(result["best_dnf"].subs({x: xv, y: yv})) == expected


def test_dnf_absorption_integration(ts, x):
    expr = sp.Or(x >= 0, sp.And(x >= 5, x <= 10))
    result = ts.simplify(expr)

    assert result["best_dnf"] == (x >= 0)
    assert_equivalent_on_grid(result["best_cnf"], x >= 0)


def test_cnf_subsumption_integration(ts, x):
    expr = sp.And(x >= 5, sp.Or(x >= 3, x < 0))
    result = ts.simplify(expr)

    assert_equivalent_on_grid(result["best_cnf"], x >= 5)
    assert_equivalent_on_grid(result["best_dnf"], x >= 5)


def test_contradictory_conjunction_integrates_to_false(ts, x):
    expr = sp.And(x > 5, x <= 5)
    result = ts.simplify(expr)

    assert result["best_dnf"] == sp.false
    assert result["best_cnf"] == sp.false


def test_tautological_disjunction_integrates_to_true(ts, x):
    expr = sp.Or(x < 3, x >= 3)
    result = ts.simplify(expr)

    assert result["best_dnf"] == sp.true
    assert result["best_cnf"] == sp.true


def test_boundary_semantics_are_preserved(ts, x):
    expr = sp.Or(x < 3, x > 3)
    result = ts.simplify(expr)

    for value, expected in [(2, True), (3, False), (4, True)]:
        assert bool(result["best_dnf"].subs(x, value)) is expected
        assert bool(result["best_cnf"].subs(x, value)) is expected


def test_closed_singleton_semantics_are_preserved(ts, x):
    expr = sp.And(x >= 3, x <= 3)
    result = ts.simplify(expr)

    for value in (2, 3, 4):
        assert bool(result["best_dnf"].subs(x, value)) is (value == 3)
        assert bool(result["best_cnf"].subs(x, value)) is (value == 3)


# ---------------------------------------------------------------------------
# Unsupported expressions and error paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        sp.Symbol("x", real=True) + 1,
        sp.Symbol("x", real=True) * 2,
        sp.Eq(sp.Symbol("x", real=True), 1),
        sp.Ne(sp.Symbol("x", real=True), 1),
    ],
)
def test_public_simplify_rejects_unsupported_non_boolean_expression(expr):
    with pytest.raises(TypeError, match="Unsupported Boolean expression"):
        ThresholdSimplifier().simplify(expr)


def test_parts_preserves_lhs_operator_rhs(ts, x):
    atom = x <= 7
    lhs, op, rhs = ts._parts(atom)
    assert lhs == x
    assert op == "<="
    assert rhs == 7


def test_invalid_internal_operator_paths_are_defensive():
    # These are intentionally malformed objects created without invoking the
    # public parser. They exercise the explicit ValueError branches.
    class FakeAtom:
        lhs = sp.Symbol("x", real=True)
        rel_op = "???"
        rhs = sp.Integer(1)

    with pytest.raises(TypeError, match=r"Not a threshold atom"):
        ThresholdSimplifier._negate_atom(FakeAtom())

    ts = ThresholdSimplifier()
    with pytest.raises(TypeError, match=r"Not a threshold atom"):
        ts._interval_implies_atom(
            [None, False, None, False],
            FakeAtom(),
        )


# ---------------------------------------------------------------------------
# Regression/property-style tests over many threshold combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lower", [-2, 0, 2])
@pytest.mark.parametrize("upper", [0, 2, 4])
@pytest.mark.parametrize("lower_closed", [False, True])
@pytest.mark.parametrize("upper_closed", [False, True])
def test_interval_validity_matches_real_interval_definition(
    ts, lower, upper, lower_closed, upper_closed
):
    interval = [
        sp.Integer(lower),
        lower_closed,
        sp.Integer(upper),
        upper_closed,
    ]
    expected = lower < upper or (lower == upper and lower_closed and upper_closed)
    assert ts._valid_interval(interval) is expected


@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
def test_negation_preserves_truth_for_boundary_values(op):
    x = sp.Symbol("x", real=True)
    atom = {
        "<": x < 3,
        "<=": x <= 3,
        ">": x > 3,
        ">=": x >= 3,
    }[op]
    negated = ThresholdSimplifier._negate_atom(atom)

    for value in (2, 3, 4):
        assert bool(atom.subs(x, value)) != bool(negated.subs(x, value))


def test_cache_is_cleared_at_start_of_simplify(ts, x):
    ts._dnf_cache[x > 0] = "stale"
    ts._cnf_cache[x > 0] = "stale"

    ts.simplify(x > 0)

    assert ts._dnf_cache.get(x > 0) != "stale"
    assert ts._cnf_cache.get(x > 0) != "stale"


def test_formula_properties(ts, x):
    # Simplified formula should be:
    # (x >= 3) & (x <= 17) & ((x > 12) | (x < 7))
    complex_formula = (
        ((x >= 3) & ((x < 7) | (x < 6)))
        | ((x > 12) & (x <= 17)) & ((x > 12) & (x <= 17))
        | ((x <= 2) & (x >= 3))
    )

    assert not is_cnf(complex_formula)
    assert not is_dnf(complex_formula)

    result = ts.simplify(complex_formula)

    assert is_cnf(result["best_cnf"])
    assert is_dnf(result["best_dnf"])

    assert equivalent(complex_formula, result["best_cnf"])
    assert equivalent(complex_formula, result["best_dnf"])


# ---------------------------------------------------------------------------
# rule_to_sympy.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def literal(index, negated=False):
    return SimpleNamespace(
        kind="Literal",
        value=index,
        negated=negated,
    )


def and_rule(*children, negated=False):
    return SimpleNamespace(
        kind="And",
        subrules=list(children),
        negated=negated,
    )


def or_rule(*children, negated=False):
    return SimpleNamespace(
        kind="Or",
        subrules=list(children),
        negated=negated,
    )


def make_clf(rule, features, **kwargs):
    return SimpleNamespace(
        rule=rule,
        feature_names=features,
        **kwargs,
    )


@pytest.fixture
def symbols():
    return {
        "mean radius": sp.Symbol("mean_radius", real=True),
        "area": sp.Symbol("area", real=True),
        "texture": sp.Symbol("texture", real=True),
    }


# ---------------------------------------------------------------------------
# Numeric and symbol-name helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("4", sp.Float("4")),
        ("13.095", sp.Float("13.095")),
        ("-2.5", sp.Float("-2.5")),
        ("+3.2", sp.Float("+3.2")),
        (".75", sp.Float(".75")),
        ("5.", sp.Float("5.")),
        ("1e3", sp.Float("1e3")),
        ("-2.5E-2", sp.Float("-2.5E-2")),
    ],
)
def test_number(value, expected):
    assert _number(value) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mean radius", "mean_radius"),
        ("  mean radius  ", "mean_radius"),
        ("feature-name", "feature_name"),
        ("feature.name", "feature_name"),
        ("a/b\\c", "a_b_c"),
        ("123feature", "_123feature"),
        ("___", None),
        ("", None),
        ("   ", None),
    ],
)
def test_safe_symbol_name(name, expected):
    if expected is None:
        with pytest.raises(ValueError, match="Invalid feature name"):
            _safe_symbol_name(name)
    else:
        assert _safe_symbol_name(name) == expected


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("mean radius", "mean radius"),
        ("mean radius=4", "mean radius"),
        ("mean radius < 5.2", "mean radius"),
        ("3.1 <= mean radius", "mean radius"),
        ("2.7 <= mean radius < 6", "mean radius"),
        (" area = -1.5 ", "area"),
        ("texture", "texture"),
    ],
)
def test_feature_base_name(feature, expected):
    assert _feature_base_name(feature) == expected


# ---------------------------------------------------------------------------
# Feature-to-SymPy conversion
# ---------------------------------------------------------------------------


def test_plain_feature_returns_symbol(symbols):
    result = feature_to_sympy("mean radius", symbols)

    assert result == symbols["mean radius"]


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("mean radius=4", sp.Eq(sp.Symbol("mean_radius", real=True), 4.0)),
        ("mean radius=-2.5", sp.Eq(sp.Symbol("mean_radius", real=True), -2.5)),
        ("mean radius=1e3", sp.Eq(sp.Symbol("mean_radius", real=True), 1000.0)),
    ],
)
def test_exact_feature(feature, expected, symbols):
    assert feature_to_sympy(feature, symbols) == expected


def test_upper_bound_feature(symbols):
    result = feature_to_sympy("mean radius < 5.2", symbols)

    assert result == (symbols["mean radius"] < sp.Float("5.2"))


def test_lower_bound_feature(symbols):
    result = feature_to_sympy("3.1 <= mean radius", symbols)

    assert result == (symbols["mean radius"] >= sp.Float("3.1"))


def test_range_feature(symbols):
    result = feature_to_sympy("2.7 <= mean radius < 6", symbols)

    expected = sp.And(
        symbols["mean radius"] >= sp.Float("2.7"),
        symbols["mean radius"] < sp.Float("6"),
    )

    assert result == expected


def test_feature_with_surrounding_whitespace(symbols):
    result = feature_to_sympy("  mean radius  ", symbols)

    assert result == symbols["mean radius"]


def test_feature_name_with_spaces(symbols):
    result = feature_to_sympy("mean radius < 10", symbols)

    assert result == (symbols["mean radius"] < sp.Float("10"))


def test_unknown_feature_raises_key_error(symbols):
    with pytest.raises(KeyError):
        feature_to_sympy("unknown feature", symbols)


# ---------------------------------------------------------------------------
# Recursive rule conversion
# ---------------------------------------------------------------------------


def test_convert_literal(symbols):
    rule = literal(0)

    result = _convert_rule(rule, ["mean radius"], symbols)

    assert result == symbols["mean radius"]


def test_convert_numpy_integer_literal(symbols):
    rule = literal(np.int64(0))

    result = _convert_rule(rule, ["mean radius"], symbols)

    assert result == symbols["mean radius"]


def test_convert_negated_literal(symbols):
    rule = literal(0, negated=True)

    result = _convert_rule(rule, ["mean radius"], symbols)

    assert result == sp.Not(symbols["mean radius"])


def test_convert_and_rule(symbols):
    rule = and_rule(literal(0), literal(1))

    result = _convert_rule(
        rule,
        ["mean radius < 5", "area"],
        symbols,
    )

    expected = sp.And(
        symbols["mean radius"] < sp.Float("5"),
        symbols["area"],
    )

    assert result == expected


def test_convert_or_rule(symbols):
    rule = or_rule(literal(0), literal(1))

    result = _convert_rule(
        rule,
        ["mean radius", "area"],
        symbols,
    )

    assert result == sp.Or(
        symbols["mean radius"],
        symbols["area"],
    )


def test_convert_nested_rule(symbols):
    rule = and_rule(
        literal(0),
        or_rule(literal(1), literal(2)),
    )

    result = _convert_rule(
        rule,
        ["mean radius", "area", "texture"],
        symbols,
    )

    expected = sp.And(
        symbols["mean radius"],
        sp.Or(symbols["area"], symbols["texture"]),
    )

    assert result == expected


def test_convert_empty_and_rule():
    assert _convert_rule(and_rule(), [], {}) is sp.true


def test_convert_empty_or_rule():
    assert _convert_rule(or_rule(), [], {}) is sp.false


def test_convert_negated_and_rule(symbols):
    rule = and_rule(literal(0), literal(1), negated=True)

    result = _convert_rule(
        rule,
        ["mean radius", "area"],
        symbols,
    )

    expected = sp.Not(sp.And(symbols["mean radius"], symbols["area"]))

    assert result == expected


def test_literal_without_value_raises_value_error():
    rule = SimpleNamespace(kind="Literal")
    rule2 = literal(None)

    with pytest.raises(ValueError, match="Literal has no value"):
        _convert_rule(rule, ["mean radius"], {})

    with pytest.raises(ValueError, match="Literal has no value"):
        _convert_rule(rule2, ["mean radius"], {})


@pytest.mark.parametrize("value", ["0", 1.5, np.int32(0)])
def test_invalid_literal_value_raises_type_error(value):
    rule = literal(value)

    with pytest.raises(TypeError, match="Literal.value must be an int"):
        _convert_rule(rule, ["mean radius", "area"], {})


@pytest.mark.parametrize("index", [-1, 1])
def test_out_of_range_literal_raises_index_error(index):
    rule = literal(index)

    with pytest.raises(IndexError, match="outside features"):
        _convert_rule(rule, ["mean radius"], {})


def test_unsupported_rule_kind_raises_type_error():
    rule = SimpleNamespace(kind="Xor")

    with pytest.raises(TypeError, match="Unsupported rule node type"):
        _convert_rule(rule, [], {})


# ---------------------------------------------------------------------------
# Public rule_to_sympy API
# ---------------------------------------------------------------------------


def test_rule_to_sympy_returns_same_classifier():
    clf = make_clf(
        literal(0),
        ["mean radius"],
    )

    result = rule_to_sympy(clf, simplify=False)

    assert result is clf


def test_rule_to_sympy_stores_intermediate_results():
    rule = literal(0)
    clf = make_clf(
        rule,
        ["mean radius"],
        format_rule=lambda: "mean radius",
    )

    rule_to_sympy(clf, simplify=False)

    assert clf.sympy.original_rule is rule
    assert clf.sympy.rule_features == ["mean radius"]
    assert clf.sympy.original_columns == ["mean radius"]
    assert clf.sympy.formatted_rule == "mean radius"
    assert clf.sympy.sympy_symbols["mean radius"] == sp.Symbol("mean_radius", real=True)
    assert clf.sympy.sympy_rule == sp.Symbol("mean_radius", real=True)
    assert clf.sympy.sympy_rule_string == "mean_radius"
    assert not hasattr(clf.sympy, "simplified_rule")


def test_rule_to_sympy_supports_features_names_attribute():
    clf = SimpleNamespace(
        rule=literal(0),
        features_names=["mean radius"],
    )

    result = rule_to_sympy(clf, simplify=False)

    assert result.sympy.rule_features == ["mean radius"]


def test_features_names_takes_precedence_over_feature_names():
    clf = SimpleNamespace(
        rule=literal(0),
        features_names=["area"],
        feature_names=["mean radius"],
    )

    rule_to_sympy(clf, simplify=False)

    assert clf.sympy.rule_features == ["area"]
    assert clf.sympy.original_columns == ["area"]


def test_missing_feature_names_raises_attribute_error():
    clf = SimpleNamespace(rule=literal(0))

    with pytest.raises(
        AttributeError,
        match="either 'features_names' or 'feature_names'",
    ):
        rule_to_sympy(clf, simplify=False)


def test_original_columns_come_from_binarizer():
    clf = make_clf(
        literal(0),
        ["mean radius < 5"],
        binarizer=SimpleNamespace(
            _original_columns=["mean radius", "area", "mean radius"]
        ),
    )

    rule_to_sympy(clf, simplify=False)

    assert clf.sympy.original_columns == ["mean radius", "area"]
    assert list(clf.sympy.sympy_symbols) == ["mean radius", "area"]


def test_original_columns_fall_back_to_feature_names():
    clf = make_clf(
        and_rule(literal(0), literal(1)),
        ["mean radius < 5", "1<=mean radius < 3"],
    )

    rule_to_sympy(clf, simplify=False)

    assert clf.sympy.original_columns == ["mean radius"]


def test_missing_format_rule_sets_formatted_rule_to_none():
    clf = make_clf(literal(0), ["mean radius"])

    rule_to_sympy(clf, simplify=False)

    assert clf.sympy.formatted_rule is None


def test_prepare_rule_is_applied_and_stored():
    clf = make_clf(literal(0), ["mean radius"])

    result = rule_to_sympy(
        clf,
        prepare_rule=lambda expr: expr > 0,
        simplify=False,
    )

    assert result.sympy.prepared_rule == (sp.Symbol("mean_radius", real=True) > 0)
    assert result.sympy.prepared_rule_string == str(result.sympy.prepared_rule)
    assert result.sympy.sympy_rule == sp.Symbol("mean_radius", real=True)


def test_prepare_rule_is_called_before_simplification():
    clf = make_clf(literal(0), ["mean radius"])
    calls = []

    def prepare(expr):
        calls.append(("prepare", expr))
        return expr

    def simplify_rule(expr):
        calls.append(("simplify", expr))
        return {"result": expr}

    rule_to_sympy(
        clf,
        prepare_rule=prepare,
        simplify_rule=simplify_rule,
        simplify=True,
    )

    assert [name for name, _ in calls] == ["prepare", "simplify"]
    assert clf.sympy.simplified_rule.result == sp.Symbol("mean_radius", real=True)


def test_custom_simplifier_is_called_when_simplify_is_true():
    clf = make_clf(literal(0), ["mean radius"])
    received = []

    def simplify_rule(expr):
        received.append(expr)
        return {"expression": expr, "status": "simplified"}

    result = rule_to_sympy(
        clf,
        simplify_rule=simplify_rule,
        simplify=True,
    )

    assert received == [sp.Symbol("mean_radius", real=True)]
    assert result.sympy.simplified_rule.expression == received[0]
    assert result.sympy.simplified_rule.status == "simplified"


def test_custom_simplifier_is_not_called_when_simplify_is_false():
    clf = make_clf(literal(0), ["mean radius"])
    called = False

    def simplify_rule(expr):
        nonlocal called
        called = True
        return {"expression": expr}

    rule_to_sympy(
        clf,
        simplify_rule=simplify_rule,
        simplify=False,
    )

    assert called is False
    assert not hasattr(clf.sympy, "simplified_rule")


def test_simplifier_receives_prepared_expression():
    clf = make_clf(literal(0), ["mean radius"])
    received = []

    def prepare(expr):
        return expr > 0

    def simplify_rule(expr):
        received.append(expr)
        return {"expression": expr}

    rule_to_sympy(
        clf,
        prepare_rule=prepare,
        simplify_rule=simplify_rule,
        simplify=True,
    )

    assert received == [sp.Symbol("mean_radius", real=True) > 0]
    assert clf.sympy.simplified_rule.expression == received[0]


def test_default_simplifier_is_not_invoked_when_disabled():
    clf = make_clf(literal(0), ["mean radius"])

    # With simplify=False, the default simplifier is configured but not called.
    result = rule_to_sympy(clf, simplify=False)

    assert result.sympy.sympy_rule == sp.Symbol("mean_radius", real=True)
    assert not hasattr(result.sympy, "simplified_rule")


if __name__ == "__main__":
    import pytest

    exit_code = pytest.main(["tests/test_postprocessing.py", "-v"])

    print(f"pytest exited with code {exit_code}")
