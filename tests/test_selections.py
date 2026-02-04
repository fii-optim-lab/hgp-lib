import doctest
import unittest
import random

import numpy as np

import hgp_lib.selections.base_selection
import hgp_lib.selections.roulette_selection
from hgp_lib.selections import BaseSelection, RouletteSelection, TournamentSelection
from hgp_lib.rules import Rule, Literal, And, Or


class TestBaseSelection(unittest.TestCase):
    def test_base_selection_is_abstract(self):
        """Test that BaseSelection cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseSelection()

    def test_base_selection_requires_select_method(self):
        """Test that subclasses must implement select method."""

        class IncompleteSelection(BaseSelection):
            pass

        with self.assertRaises(TypeError):
            IncompleteSelection()

    def test_base_selection_with_implementation(self):
        """Test that a proper subclass can be instantiated and used."""

        class TopSelection(BaseSelection):
            def select(self, rules, scores, n_select):
                scores = np.asarray(scores)
                ranked = sorted(zip(scores, rules), reverse=True)
                selected_rules = [rule.copy() for _, rule in ranked[:n_select]]
                selected_scores = np.array([s for s, _ in ranked[:n_select]])
                return selected_rules, selected_scores

        selection = TopSelection()
        rules = [Literal(value=0), Literal(value=1), Literal(value=2)]
        scores = [0.3, 0.9, 0.5]
        selected_rules, selected_scores = selection.select(rules, scores, 2)

        self.assertEqual(len(selected_rules), 2)
        self.assertIsInstance(selected_rules[0], Rule)
        self.assertIsInstance(selected_rules[1], Rule)
        # Should select the top 2 (scores 0.9 and 0.5)
        self.assertEqual(selected_rules[0].value, 1)  # score 0.9
        self.assertEqual(selected_rules[1].value, 2)  # score 0.5
        # Scores should be returned
        self.assertEqual(len(selected_scores), 2)
        self.assertAlmostEqual(selected_scores[0], 0.9)
        self.assertAlmostEqual(selected_scores[1], 0.5)

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.selections.base_selection, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


class TestRouletteSelection(unittest.TestCase):
    def test_select_with_replacement(self):
        """Test that selection with replacement works correctly."""
        selection = RouletteSelection()
        rules = [
            Literal(value=0),
            Literal(value=1),
            Literal(value=2),
        ]
        scores = [0.1, 0.5, 0.4]

        random.seed(42)
        np.random.seed(42)
        selected_rules, selected_scores = selection.select(rules, scores, n_select=5)

        # Should return 5 rules (can have duplicates)
        self.assertEqual(len(selected_rules), 5)
        self.assertEqual(len(selected_scores), 5)
        # All should be Rule instances
        self.assertTrue(all(isinstance(rule, Rule) for rule in selected_rules))

    def test_select_handles_negative_scores(self):
        """Test that negative scores are handled by shifting."""
        selection = RouletteSelection()
        rules = [Literal(value=0), Literal(value=1), Literal(value=2)]
        scores = [-0.5, 0.3, -0.1]

        random.seed(42)
        np.random.seed(42)
        selected_rules, selected_scores = selection.select(rules, scores, n_select=2)

        # Should still work and return 2 rules
        self.assertEqual(len(selected_rules), 2)
        self.assertEqual(len(selected_scores), 2)
        self.assertTrue(all(isinstance(rule, Rule) for rule in selected_rules))

    def test_select_handles_equal_scores(self):
        """Test that equal scores result in uniform selection."""
        selection = RouletteSelection()
        rules = [
            Literal(value=0),
            Literal(value=1),
            Literal(value=2),
        ]
        scores = [0.5, 0.5, 0.5]

        random.seed(42)
        np.random.seed(42)
        selected_rules, selected_scores = selection.select(rules, scores, n_select=3)

        # Should return 3 rules (all should be selectable)
        self.assertEqual(len(selected_rules), 3)
        self.assertEqual(len(selected_scores), 3)

    def test_select_handles_zero_scores(self):
        """Test that zero scores are handled correctly."""
        selection = RouletteSelection()
        rules = [Literal(value=0), Literal(value=1), Literal(value=2)]
        scores = [0.0, 0.0, 0.0]

        random.seed(42)
        np.random.seed(42)
        selected_rules, selected_scores = selection.select(rules, scores, n_select=2)

        # Should use uniform distribution
        self.assertEqual(len(selected_rules), 2)
        self.assertEqual(len(selected_scores), 2)
        self.assertTrue(all(isinstance(rule, Rule) for rule in selected_rules))

    def test_select_single_rule(self):
        """Test selection when only one rule is requested."""
        selection = RouletteSelection()
        rules = [Literal(value=0), Literal(value=1), Literal(value=2)]
        scores = [0.1, 0.5, 0.4]

        random.seed(42)
        np.random.seed(42)
        selected_rules, selected_scores = selection.select(rules, scores, n_select=1)

        self.assertEqual(len(selected_rules), 1)
        self.assertEqual(len(selected_scores), 1)
        self.assertIsInstance(selected_rules[0], Rule)

    def test_select_empty_list(self):
        """Test that selecting from empty list returns empty list."""
        selection = RouletteSelection()
        selected_rules, selected_scores = selection.select([], [], n_select=1)
        self.assertEqual(selected_rules, [])
        self.assertEqual(len(selected_scores), 0)

    def test_select_probability_proportionality(self):
        """Test that selection probability is proportional to fitness."""
        selection = RouletteSelection()
        rules = [
            Literal(value=0),  # Low fitness
            Literal(value=1),  # High fitness
        ]
        scores = [0.1, 0.9]  # Rule 1 should be selected much more often

        random.seed(42)
        np.random.seed(42)
        # Select many times to test probability
        all_selected = []
        for _ in range(1000):
            selected_rules, _ = selection.select(rules, scores, n_select=1)
            all_selected.append(selected_rules[0].value)

        # Count occurrences
        count_0 = all_selected.count(0)
        count_1 = all_selected.count(1)

        # Rule 1 (score 0.9) should be selected much more often than rule 0 (score 0.1)
        # With 1000 samples, we expect ~900 selections of rule 1
        # Allow some variance but should be clearly biased
        self.assertGreater(count_1, count_0)
        self.assertGreater(count_1, 800)  # Should be selected at least 80% of the time

    def test_select_with_complex_rules(self):
        """Test selection with complex rule structures."""
        selection = RouletteSelection()
        rules = [
            And([Literal(value=0), Literal(value=1)]),
            Or([Literal(value=2), Literal(value=3)]),
            And([Literal(value=4), Literal(value=5)]),
        ]
        scores = [0.2, 0.6, 0.2]

        random.seed(42)
        np.random.seed(42)
        selected_rules, selected_scores = selection.select(rules, scores, n_select=2)

        self.assertEqual(len(selected_rules), 2)
        self.assertEqual(len(selected_scores), 2)
        for rule in selected_rules:
            self.assertIsInstance(rule, Rule)

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.selections.roulette_selection, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


class TestTournamentSelection(unittest.TestCase):
    def test_init_default_parameters(self):
        """Test that default parameters are set correctly."""
        selection = TournamentSelection()
        self.assertEqual(selection.tournament_size, 10)
        self.assertEqual(selection.selection_p, 0.4)

    def test_init_custom_parameters(self):
        """Test that custom parameters are set correctly."""
        selection = TournamentSelection(tournament_size=5, selection_p=0.6)
        self.assertEqual(selection.tournament_size, 5)
        self.assertEqual(selection.selection_p, 0.6)

    def test_init_invalid_tournament_size_type(self):
        """Test that non-integer tournament_size raises TypeError."""
        with self.assertRaises(TypeError):
            TournamentSelection(tournament_size=5.0)

    def test_init_invalid_selection_p_type(self):
        """Test that non-float selection_p raises TypeError."""
        with self.assertRaises(TypeError):
            TournamentSelection(selection_p=1)

    def test_init_tournament_size_zero(self):
        """Test that tournament_size of 0 raises ValueError."""
        with self.assertRaises(ValueError):
            TournamentSelection(tournament_size=0)

    def test_init_tournament_size_negative(self):
        """Test that negative tournament_size raises ValueError."""
        with self.assertRaises(ValueError):
            TournamentSelection(tournament_size=-1)

    def test_init_selection_p_negative(self):
        """Test that negative selection_p raises ValueError."""
        with self.assertRaises(ValueError):
            TournamentSelection(selection_p=-0.1)

    def test_init_selection_p_greater_than_one(self):
        """Test that selection_p > 1 raises ValueError."""
        with self.assertRaises(ValueError):
            TournamentSelection(selection_p=1.1)

    def test_init_selection_p_boundary_zero(self):
        """Test that selection_p of 0 is valid."""
        selection = TournamentSelection(selection_p=0.0)
        self.assertEqual(selection.selection_p, 0.0)

    def test_init_selection_p_boundary_one(self):
        """Test that selection_p of 1 is valid."""
        selection = TournamentSelection(selection_p=1.0)
        self.assertEqual(selection.selection_p, 1.0)

    def test_probs_precomputed(self):
        """Test that probability array is precomputed correctly."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        # P(0) = 0.5, P(1) = 0.25, P(2) = 0.25 (catches remainder)
        expected = np.array([0.5, 0.25, 0.25])
        np.testing.assert_array_almost_equal(selection.probs, expected)

    def test_probs_sum_to_one(self):
        """Test that precomputed probabilities sum to 1."""
        selection = TournamentSelection(tournament_size=10, selection_p=0.4)
        self.assertAlmostEqual(np.sum(selection.probs), 1.0)

    def test_select_returns_correct_count(self):
        """Test that select returns the correct number of rules."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        rules = [Literal(value=i) for i in range(5)]
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        np.random.seed(42)
        selected_rules, selected_scores = selection.select(rules, scores, n_select=3)
        self.assertEqual(len(selected_rules), 3)
        self.assertEqual(len(selected_scores), 3)

    def test_select_returns_rule_instances(self):
        """Test that all selected items are Rule instances."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        rules = [Literal(value=i) for i in range(5)]
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        np.random.seed(42)
        selected_rules, _ = selection.select(rules, scores, n_select=3)
        self.assertTrue(all(isinstance(rule, Rule) for rule in selected_rules))

    def test_select_returns_copies(self):
        """Test that selected rules are copies, not originals."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        rules = [Literal(value=i) for i in range(5)]
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        np.random.seed(42)
        selected_rules, _ = selection.select(rules, scores, n_select=3)
        for selected_rule in selected_rules:
            for original_rule in rules:
                self.assertIsNot(selected_rule, original_rule)

    def test_select_with_replacement(self):
        """Test that the same rule can appear multiple times."""
        selection = TournamentSelection(tournament_size=3, selection_p=1.0)
        rules = [Literal(value=i) for i in range(5)]
        # Make one rule clearly the best
        scores = [0.1, 0.9, 0.2, 0.15, 0.1]
        np.random.seed(42)
        # With selection_p=1.0, best in each tournament always wins
        selected_rules, _ = selection.select(rules, scores, n_select=10)
        values = [r.value for r in selected_rules]
        # The best rule (value=1) should appear multiple times
        self.assertGreater(values.count(1), 1)

    def test_select_single_rule(self):
        """Test selection when only one rule is requested."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        rules = [Literal(value=i) for i in range(5)]
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        np.random.seed(42)
        selected_rules, selected_scores = selection.select(rules, scores, n_select=1)
        self.assertEqual(len(selected_rules), 1)
        self.assertEqual(len(selected_scores), 1)
        self.assertIsInstance(selected_rules[0], Rule)

    def test_select_all_rules(self):
        """Test selection when n_select equals population size."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        rules = [Literal(value=i) for i in range(5)]
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        np.random.seed(42)
        selected_rules, _ = selection.select(rules, scores, n_select=5)
        self.assertEqual(len(selected_rules), 5)

    def test_select_favors_higher_fitness(self):
        """Test that higher fitness rules are selected more often."""
        selection = TournamentSelection(tournament_size=2, selection_p=0.7)
        rules = [Literal(value=0), Literal(value=1)]
        scores = [0.1, 0.9]  # Rule 1 has much higher fitness
        np.random.seed(42)
        # Select many times to test probability
        count_0 = 0
        count_1 = 0
        for _ in range(1000):
            selected_rules, _ = selection.select(rules, scores, n_select=1)
            if selected_rules[0].value == 0:
                count_0 += 1
            else:
                count_1 += 1
        # Rule 1 should be selected more often
        self.assertGreater(count_1, count_0)
        self.assertGreater(count_1, 600)  # Should be selected majority of the time

    def test_select_with_equal_scores(self):
        """Test selection when all scores are equal."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        rules = [Literal(value=i) for i in range(5)]
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        np.random.seed(42)
        selected_rules, _ = selection.select(rules, scores, n_select=3)
        self.assertEqual(len(selected_rules), 3)
        self.assertTrue(all(isinstance(rule, Rule) for rule in selected_rules))

    def test_select_with_negative_scores(self):
        """Test selection with negative fitness scores."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        rules = [Literal(value=i) for i in range(5)]
        scores = [-0.5, 0.3, -0.1, 0.0, 0.2]
        np.random.seed(42)
        selected_rules, _ = selection.select(rules, scores, n_select=3)
        self.assertEqual(len(selected_rules), 3)
        self.assertTrue(all(isinstance(rule, Rule) for rule in selected_rules))

    def test_select_with_zero_scores(self):
        """Test selection when all scores are zero."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        rules = [Literal(value=i) for i in range(5)]
        scores = [0.0, 0.0, 0.0, 0.0, 0.0]
        np.random.seed(42)
        selected_rules, _ = selection.select(rules, scores, n_select=3)
        self.assertEqual(len(selected_rules), 3)

    def test_select_with_selection_p_one(self):
        """Test that selection_p=1.0 always selects the best in tournament."""
        selection = TournamentSelection(tournament_size=5, selection_p=1.0)
        rules = [Literal(value=i) for i in range(10)]
        # Clear ranking: rule 0 is best, rule 9 is worst
        scores = list(range(10, 0, -1))
        np.random.seed(3)
        # With selection_p=1.0, always picks best in tournament
        selected_rules, _ = selection.select(rules, scores, n_select=100)
        # All selected should be among the top candidates
        values = [r.value for r in selected_rules]
        # Best rule (value=0) should dominate
        self.assertGreater(values.count(0), 50)

    def test_select_with_selection_p_zero(self):
        """Test that selection_p=0.0 always selects the worst in tournament."""
        selection = TournamentSelection(tournament_size=5, selection_p=0.0)
        rules = [Literal(value=i) for i in range(10)]
        scores = list(range(10, 0, -1))  # rule 0 is best
        np.random.seed(42)
        # With selection_p=0.0, all probability goes to last rank
        selected_rules, _ = selection.select(rules, scores, n_select=100)
        values = [r.value for r in selected_rules]
        # Worst candidates should dominate
        worst_count = sum(1 for v in values if v >= 5)
        self.assertGreater(worst_count, 50)

    def test_select_with_complex_rules(self):
        """Test selection with complex rule structures."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        rules = [
            And([Literal(value=0), Literal(value=1)]),
            Or([Literal(value=2), Literal(value=3)]),
            And([Literal(value=4), Literal(value=5)]),
        ]
        scores = [0.2, 0.6, 0.2]
        np.random.seed(42)
        selected_rules, _ = selection.select(rules, scores, n_select=2)
        self.assertEqual(len(selected_rules), 2)
        for rule in selected_rules:
            self.assertIsInstance(rule, Rule)

    def test_select_with_numpy_scores(self):
        """Test selection with numpy array scores."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        rules = [Literal(value=i) for i in range(5)]
        scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        np.random.seed(42)
        selected_rules, _ = selection.select(rules, scores, n_select=3)
        self.assertEqual(len(selected_rules), 3)

    def test_select_with_list_scores(self):
        """Test selection with list scores."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        rules = [Literal(value=i) for i in range(5)]
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        np.random.seed(42)
        selected_rules, _ = selection.select(rules, scores, n_select=3)
        self.assertEqual(len(selected_rules), 3)

    def test_select_deterministic_with_seed(self):
        """Test that selection is deterministic with fixed seed."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        rules = [Literal(value=i) for i in range(5)]
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]

        np.random.seed(42)
        selected1, _ = selection.select(rules, scores, n_select=3)
        values1 = [r.value for r in selected1]

        np.random.seed(42)
        selected2, _ = selection.select(rules, scores, n_select=3)
        values2 = [r.value for r in selected2]

        self.assertEqual(values1, values2)

    def test_select_tournament_size_equals_population(self):
        """Test when tournament size equals population size."""
        selection = TournamentSelection(tournament_size=5, selection_p=0.5)
        rules = [Literal(value=i) for i in range(5)]
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        np.random.seed(42)
        selected_rules, _ = selection.select(rules, scores, n_select=3)
        self.assertEqual(len(selected_rules), 3)

    def test_select_large_n_select(self):
        """Test selection with n_select larger than population."""
        selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        rules = [Literal(value=i) for i in range(5)]
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        np.random.seed(42)
        selected_rules, _ = selection.select(rules, scores, n_select=20)
        self.assertEqual(len(selected_rules), 20)

    def test_doctests(self):
        """Test all doctests in the tournament_selection module."""
        result = doctest.testmod(hgp_lib.selections.tournament_selection, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
