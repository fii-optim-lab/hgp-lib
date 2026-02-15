"""Unit tests for GenerationMetrics construction."""

import unittest

from hgp_lib.metrics.core import GenerationMetrics
from hgp_lib.rules import Literal, And


class TestGenerationMetrics(unittest.TestCase):
    """Tests for GenerationMetrics construction and properties."""

    def test_from_population_basic(self):
        """Test basic construction computes all derived fields correctly."""
        best_rule = And([Literal(value=1), Literal(value=2)])
        train_scores = [0.7, 0.9]
        complexities = [1, 3]

        metrics = GenerationMetrics.from_population(
            best_idx=1,
            best_rule=best_rule,
            train_scores=train_scores,
            complexities=complexities,
            child_population_generation_metrics=[],
        )

        self.assertEqual(metrics.population_size, 2)
        self.assertEqual(metrics.complexities, [1, 3])
        self.assertEqual(str(metrics.best_rule), "And(1, 2)")
        self.assertAlmostEqual(metrics.best_train_score, 0.9)

    def test_from_population_with_val_score(self):
        """Test construction with validation score via replace."""
        from dataclasses import replace

        best_rule = Literal(value=0)
        metrics = GenerationMetrics.from_population(
            best_idx=0,
            best_rule=best_rule,
            train_scores=[0.8],
            complexities=[1],
            child_population_generation_metrics=[],
        )

        metrics_with_val = replace(metrics, val_score=0.6)
        self.assertEqual(metrics_with_val.val_score, 0.6)
        self.assertIsNone(metrics.val_score)

    def test_hierarchical_metrics(self):
        """Test construction with child population metrics."""
        child = GenerationMetrics.from_population(
            best_idx=0,
            best_rule=Literal(value=0),
            train_scores=[0.5],
            complexities=[1],
            child_population_generation_metrics=[],
        )

        metrics = GenerationMetrics.from_population(
            best_idx=0,
            best_rule=Literal(value=0),
            train_scores=[0.7],
            complexities=[1],
            child_population_generation_metrics=[child],
        )

        self.assertEqual(len(metrics.child_population_generation_metrics), 1)

    def test_best_train_score(self):
        """Test best_train_score property."""
        metrics = GenerationMetrics.from_population(
            best_idx=1,
            best_rule=Literal(value=1),
            train_scores=[0.3, 0.9, 0.5],
            complexities=[1, 3, 2],
            child_population_generation_metrics=[],
        )
        self.assertAlmostEqual(metrics.best_train_score, 0.9)

    def test_best_rule_complexity(self):
        """Test best_rule_complexity property."""
        metrics = GenerationMetrics.from_population(
            best_idx=1,
            best_rule=Literal(value=1),
            train_scores=[0.3, 0.9],
            complexities=[1, 5],
            child_population_generation_metrics=[],
        )
        self.assertEqual(metrics.best_rule_complexity, 5)


if __name__ == "__main__":
    unittest.main()
