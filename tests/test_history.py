"""Unit tests for PopulationHistory."""

import unittest
from dataclasses import replace

from hgp_lib.metrics.history import PopulationHistory
from hgp_lib.metrics.core import GenerationMetrics
from hgp_lib.rules import Literal


class TestPopulationHistory(unittest.TestCase):
    """Tests for PopulationHistory dataclass."""

    def test_empty_history(self):
        """Test empty history has zero generations."""
        ph = PopulationHistory(global_best_rule=Literal(value=0))
        self.assertEqual(len(ph.generations), 0)

    def test_with_generations(self):
        """Test history with generations."""
        gen = GenerationMetrics.from_population(
            generation=0,
            best_idx=0,
            best_rule=Literal(value=0),
            train_scores=[0.8],
            complexities=[1],
            child_population_generation_metrics=[],
        )
        ph = PopulationHistory(
            global_best_rule=Literal(value=0),
            generations=[gen],
        )
        self.assertEqual(len(ph.generations), 1)

    def test_best_val_score_none_when_no_val(self):
        """Test best_val_score is None when no validation scores exist."""
        gen = GenerationMetrics.from_population(
            generation=0,
            best_idx=0,
            best_rule=Literal(value=0),
            train_scores=[0.8],
            complexities=[1],
            child_population_generation_metrics=[],
        )
        ph = PopulationHistory(
            global_best_rule=Literal(value=0),
            generations=[gen],
        )
        self.assertIsNone(ph.best_val_score)

    def test_best_val_score_with_val(self):
        """Test best_val_score returns max val score."""
        gen1 = GenerationMetrics.from_population(
            generation=0,
            best_idx=0,
            best_rule=Literal(value=0),
            train_scores=[0.8],
            complexities=[1],
            child_population_generation_metrics=[],
        )
        gen1 = replace(gen1, val_score=0.6)

        gen2 = GenerationMetrics.from_population(
            generation=1,
            best_idx=0,
            best_rule=Literal(value=0),
            train_scores=[0.9],
            complexities=[1],
            child_population_generation_metrics=[],
        )
        gen2 = replace(gen2, val_score=0.75)

        ph = PopulationHistory(
            global_best_rule=Literal(value=0),
            generations=[gen1, gen2],
        )
        self.assertAlmostEqual(ph.best_val_score, 0.75)


if __name__ == "__main__":
    unittest.main()
