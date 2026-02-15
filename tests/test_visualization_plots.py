"""Unit tests for visualization plot functions."""

import unittest
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")

from hgp_lib.metrics.core import GenerationMetrics
from hgp_lib.metrics.history import PopulationHistory
from hgp_lib.metrics.results import RunResult, ExperimentResult
from hgp_lib.metrics.visualization.plots import (
    plot_experiment_boxplots,
    plot_best_fold_generations,
    plot_all_folds_val_scores,
)
from hgp_lib.rules import Literal


def _create_generation_metrics(generation: int, score: float, val_score: float | None = None) -> GenerationMetrics:
    """Helper to create a simple GenerationMetrics."""
    gen = GenerationMetrics.from_population(
        generation=generation,
        best_idx=0,
        best_rule=Literal(value=0),
        train_scores=[score],
        complexities=[1],
        child_population_generation_metrics=[],
    )
    if val_score is not None:
        gen = replace(gen, val_score=val_score)
    return gen


def _create_experiment() -> ExperimentResult:
    """Create a minimal ExperimentResult for testing."""
    runs = []
    for run_id in range(3):
        folds = []
        for _ in range(2):
            gens = [
                _create_generation_metrics(i, 0.5 + i * 0.05, val_score=0.4 + i * 0.03)
                for i in range(5)
            ]
            fold = PopulationHistory(
                global_best_rule=Literal(value=0),
                generations=gens,
            )
            folds.append(fold)
        runs.append(RunResult(
            run_id=run_id,
            seed=run_id,
            best_fold_idx=0,
            folds=folds,
            test_score=0.6 + run_id * 0.05,
            feature_names={0: "f0"},
        ))
    return ExperimentResult(runs=runs)


class TestPlots(unittest.TestCase):
    """Tests that all plot functions return figures without errors."""

    def test_experiment_boxplots(self):
        experiment = _create_experiment()
        fig = plot_experiment_boxplots(experiment)
        self.assertIsNotNone(fig)

    def test_best_fold_generations(self):
        experiment = _create_experiment()
        fig = plot_best_fold_generations(experiment)
        self.assertIsNotNone(fig)

    def test_all_folds_val_scores(self):
        experiment = _create_experiment()
        fig = plot_all_folds_val_scores(experiment)
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
