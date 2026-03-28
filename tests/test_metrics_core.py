"""Unit tests for GenerationMetrics."""

import doctest
import unittest
from dataclasses import replace

import hgp_lib.metrics.core
from hgp_lib.metrics.core import GenerationMetrics
from hgp_lib.rules import Literal, And


class TestGenerationMetrics(unittest.TestCase):
    def _make(self, **kwargs):
        defaults = dict(
            best_idx=0,
            best_rule=Literal(value=0),
            train_scores=[0.8],
            complexities=[1],
            child_population_generation_metrics=[],
        )
        defaults.update(kwargs)
        return GenerationMetrics.from_population(**defaults)

    # ------------------------------------------------------------------ #
    #  from_population
    # ------------------------------------------------------------------ #
    def test_from_population_basic(self):
        m = self._make(
            best_idx=1,
            best_rule=And([Literal(value=1), Literal(value=2)]),
            train_scores=[0.7, 0.9],
            complexities=[1, 3],
        )
        self.assertEqual(m.population_size, 2)
        self.assertAlmostEqual(m.best_train_score, 0.9)
        self.assertEqual(str(m.best_rule), "And(1, 2)")

    def test_from_population_val_score_is_none(self):
        m = self._make()
        self.assertIsNone(m.val_score)

    def test_from_population_with_val_score(self):
        m = replace(self._make(), val_score=0.6)
        self.assertEqual(m.val_score, 0.6)

    # ------------------------------------------------------------------ #
    #  best_train_score
    # ------------------------------------------------------------------ #
    def test_best_train_score(self):
        m = self._make(best_idx=2, train_scores=[0.3, 0.5, 0.9], complexities=[1, 2, 3])
        self.assertAlmostEqual(m.best_train_score, 0.9)

    def test_best_train_score_first(self):
        m = self._make(best_idx=0, train_scores=[1.0, 0.5])
        self.assertAlmostEqual(m.best_train_score, 1.0)

    # ------------------------------------------------------------------ #
    #  best_rule_complexity
    # ------------------------------------------------------------------ #
    def test_best_rule_complexity(self):
        m = self._make(best_idx=1, train_scores=[0.3, 0.9], complexities=[1, 5])
        self.assertEqual(m.best_rule_complexity, 5)

    def test_best_rule_complexity_single(self):
        m = self._make(best_idx=0, complexities=[7])
        self.assertEqual(m.best_rule_complexity, 7)

    # ------------------------------------------------------------------ #
    #  population_size
    # ------------------------------------------------------------------ #
    def test_population_size(self):
        m = self._make(train_scores=[0.1, 0.2, 0.3], complexities=[1, 2, 3])
        self.assertEqual(m.population_size, 3)

    def test_population_size_single(self):
        m = self._make()
        self.assertEqual(m.population_size, 1)

    # ------------------------------------------------------------------ #
    #  child_population_generation_metrics
    # ------------------------------------------------------------------ #
    def test_hierarchical_metrics(self):
        child = self._make(train_scores=[0.5])
        parent = self._make(
            train_scores=[0.7],
            child_population_generation_metrics=[child],
        )
        self.assertEqual(len(parent.child_population_generation_metrics), 1)
        self.assertAlmostEqual(
            parent.child_population_generation_metrics[0].best_train_score,
            0.5,
        )

    def test_empty_children(self):
        m = self._make()
        self.assertEqual(len(m.child_population_generation_metrics), 0)

    # ------------------------------------------------------------------ #
    #  Doctests
    # ------------------------------------------------------------------ #
    def test_doctests(self):
        result = doctest.testmod(hgp_lib.metrics.core, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
