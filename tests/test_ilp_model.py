"""Tests for the ILP model (hgp_lib.populations.ilp_model)."""

import unittest
import numpy as np

from hgp_lib.populations.ilp_model import solve_best_rule_ilp, ILPSolution


class TestSolveBestRuleILP(unittest.TestCase):
    """Tests for solve_best_rule_ilp."""

    def setUp(self):
        np.random.seed(42)

    # ------------------------------------------------------------------
    # Basic contract
    # ------------------------------------------------------------------

    def test_returns_ilp_solution(self):
        """solve_best_rule_ilp returns an ILPSolution dataclass."""
        data = np.array([[True, False], [False, True], [True, True], [False, False]])
        labels = np.array([1, 0, 1, 0])
        sol = solve_best_rule_ilp(data, labels, use_and=True)
        self.assertIsInstance(sol, ILPSolution)

    def test_feasible_solution_has_selected_features(self):
        """A feasible solution selects at least min_literals features."""
        data = np.random.rand(30, 6) > 0.5
        labels = np.random.randint(0, 2, 30)
        sol = solve_best_rule_ilp(data, labels, use_and=False, min_literals=2)
        if sol.feasible:
            self.assertGreaterEqual(len(sol.selected_features), 2)

    def test_selected_features_within_bounds(self):
        """Selected features are in [0, n_features)."""
        data = np.random.rand(40, 8) > 0.5
        labels = np.random.randint(0, 2, 40)
        sol = solve_best_rule_ilp(data, labels, use_and=True, max_literals=4)
        if sol.feasible:
            self.assertTrue(np.all(sol.selected_features >= 0))
            self.assertTrue(np.all(sol.selected_features < 8))

    def test_negations_same_length_as_selected(self):
        """negations array has the same length as selected_features."""
        data = np.random.rand(30, 5) > 0.5
        labels = np.random.randint(0, 2, 30)
        sol = solve_best_rule_ilp(data, labels, use_and=False)
        self.assertEqual(len(sol.selected_features), len(sol.negations))

    def test_cardinality_upper_bound(self):
        """Number of selected literals does not exceed max_literals."""
        data = np.random.rand(40, 10) > 0.5
        labels = np.random.randint(0, 2, 40)
        sol = solve_best_rule_ilp(data, labels, use_and=True, max_literals=3)
        if sol.feasible:
            self.assertLessEqual(len(sol.selected_features), 3)

    def test_cardinality_lower_bound(self):
        """Number of selected literals is at least min_literals."""
        data = np.random.rand(40, 10) > 0.5
        labels = np.random.randint(0, 2, 40)
        sol = solve_best_rule_ilp(data, labels, use_and=False, min_literals=3)
        if sol.feasible:
            self.assertGreaterEqual(len(sol.selected_features), 3)

    def test_no_duplicate_features(self):
        """Each feature index appears at most once (one polarity constraint)."""
        data = np.random.rand(40, 8) > 0.5
        labels = np.random.randint(0, 2, 40)
        sol = solve_best_rule_ilp(data, labels, use_and=True)
        if sol.feasible:
            self.assertEqual(
                len(sol.selected_features), len(set(sol.selected_features.tolist()))
            )

    # ------------------------------------------------------------------
    # Operator semantics
    # ------------------------------------------------------------------

    def test_use_and_flag_preserved(self):
        """The use_and flag in the solution matches the input."""
        data = np.random.rand(20, 4) > 0.5
        labels = np.random.randint(0, 2, 20)
        for use_and in (True, False):
            sol = solve_best_rule_ilp(data, labels, use_and=use_and)
            self.assertEqual(sol.use_and, use_and)

    def test_and_rule_finds_known_pattern(self):
        """AND solver recovers a known AND pattern: label = feat0 AND feat1."""
        np.random.seed(0)
        data = np.random.rand(60, 5) > 0.5
        labels = (data[:, 0] & data[:, 1]).astype(int)
        sol = solve_best_rule_ilp(
            data, labels, use_and=True, min_literals=2, max_literals=3
        )
        self.assertTrue(sol.feasible)
        # Features 0 and 1 should be selected (positive)
        sel = set(sol.selected_features.tolist())
        self.assertIn(0, sel)
        self.assertIn(1, sel)
        # Both should be non-negated
        for i, feat in enumerate(sol.selected_features):
            if feat in (0, 1):
                self.assertFalse(sol.negations[i])

    def test_or_rule_finds_known_pattern(self):
        """OR solver recovers a known OR pattern: label = feat0 OR feat2."""
        np.random.seed(0)
        data = np.random.rand(60, 5) > 0.5
        labels = (data[:, 0] | data[:, 2]).astype(int)
        sol = solve_best_rule_ilp(
            data, labels, use_and=False, min_literals=2, max_literals=3
        )
        self.assertTrue(sol.feasible)
        sel = set(sol.selected_features.tolist())
        self.assertIn(0, sel)
        self.assertIn(2, sel)
        for i, feat in enumerate(sol.selected_features):
            if feat in (0, 2):
                self.assertFalse(sol.negations[i])

    # ------------------------------------------------------------------
    # Infeasible / edge cases
    # ------------------------------------------------------------------

    def test_infeasible_when_min_exceeds_features(self):
        """Infeasible when min_literals > number of features."""
        data = np.random.rand(10, 2) > 0.5
        labels = np.random.randint(0, 2, 10)
        sol = solve_best_rule_ilp(data, labels, use_and=True, min_literals=5, max_literals=5)
        self.assertFalse(sol.feasible)

    def test_empty_arrays_on_infeasible(self):
        """Infeasible solution returns empty arrays."""
        data = np.random.rand(10, 2) > 0.5
        labels = np.random.randint(0, 2, 10)
        sol = solve_best_rule_ilp(data, labels, use_and=True, min_literals=5, max_literals=5)
        self.assertEqual(len(sol.selected_features), 0)
        self.assertEqual(len(sol.negations), 0)


if __name__ == "__main__":
    unittest.main()
