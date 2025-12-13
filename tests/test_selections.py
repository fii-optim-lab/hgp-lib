import doctest
import unittest
import random

import numpy as np

import hgp_lib.selections.base_selection
import hgp_lib.selections.roulette_selection
from hgp_lib.selections import BaseSelection, RouletteSelection
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
                ranked = sorted(zip(scores, rules), reverse=True)
                return [rule.copy() for _, rule in ranked[:n_select]]

        selection = TopSelection()
        rules = [Literal(value=0), Literal(value=1), Literal(value=2)]
        scores = [0.3, 0.9, 0.5]
        selected = selection.select(rules, scores, 2)

        self.assertEqual(len(selected), 2)
        self.assertIsInstance(selected[0], Rule)
        self.assertIsInstance(selected[1], Rule)
        # Should select the top 2 (scores 0.9 and 0.5)
        self.assertEqual(selected[0].value, 1)  # score 0.9
        self.assertEqual(selected[1].value, 2)  # score 0.5

    def test_doctests(self):
        result = doctest.testmod(
            hgp_lib.selections.base_selection, verbose=False
        )
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
        selected = selection.select(rules, scores, n_select=5)

        # Should return 5 rules (can have duplicates)
        self.assertEqual(len(selected), 5)
        # All should be Rule instances
        self.assertTrue(all(isinstance(rule, Rule) for rule in selected))

    def test_select_handles_negative_scores(self):
        """Test that negative scores are handled by shifting."""
        selection = RouletteSelection()
        rules = [Literal(value=0), Literal(value=1), Literal(value=2)]
        scores = [-0.5, 0.3, -0.1]

        random.seed(42)
        np.random.seed(42)
        selected = selection.select(rules, scores, n_select=2)

        # Should still work and return 2 rules
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(isinstance(rule, Rule) for rule in selected))

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
        selected = selection.select(rules, scores, n_select=3)

        # Should return 3 rules (all should be selectable)
        self.assertEqual(len(selected), 3)

    def test_select_handles_zero_scores(self):
        """Test that zero scores are handled correctly."""
        selection = RouletteSelection()
        rules = [Literal(value=0), Literal(value=1), Literal(value=2)]
        scores = [0.0, 0.0, 0.0]

        random.seed(42)
        np.random.seed(42)
        selected = selection.select(rules, scores, n_select=2)

        # Should use uniform distribution
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(isinstance(rule, Rule) for rule in selected))

    def test_select_single_rule(self):
        """Test selection when only one rule is requested."""
        selection = RouletteSelection()
        rules = [Literal(value=0), Literal(value=1), Literal(value=2)]
        scores = [0.1, 0.5, 0.4]

        random.seed(42)
        np.random.seed(42)
        selected = selection.select(rules, scores, n_select=1)

        self.assertEqual(len(selected), 1)
        self.assertIsInstance(selected[0], Rule)

    def test_select_empty_list(self):
        """Test that selecting from empty list returns empty list."""
        selection = RouletteSelection()
        selected = selection.select([], [], n_select=1)
        self.assertEqual(selected, [])

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
            selected = selection.select(rules, scores, n_select=1)
            all_selected.append(selected[0].value)

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
        selected = selection.select(rules, scores, n_select=2)

        self.assertEqual(len(selected), 2)
        for rule in selected:
            self.assertIsInstance(rule, Rule)

    def test_doctests(self):
        result = doctest.testmod(
            hgp_lib.selections.roulette_selection, verbose=False
        )
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
