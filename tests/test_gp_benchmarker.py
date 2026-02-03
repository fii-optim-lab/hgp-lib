import doctest
import unittest
import random

import numpy as np

import hgp_lib
from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.mutations import (
    MutationExecutor,
    create_standard_literal_mutations,
    create_standard_operator_mutations,
)
from hgp_lib.populations import PopulationGenerator, RandomStrategy
from hgp_lib.rules import Rule
from hgp_lib.selections import RouletteSelection


def accuracy(predictions, labels):
    """Module-level score function for pickling in parallel tests."""
    return np.mean(predictions == labels)


def accuracy_with_sample_weight(predictions, labels, sample_weight=None):
    """Module-level score function that supports sample_weight for optimization tests."""
    if sample_weight is None:
        return np.mean(predictions == labels)
    correct = predictions == labels
    return np.dot(correct, sample_weight) / sample_weight.sum()


class TestGPBenchmarker(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        np.random.seed(42)

        self.data = np.array(
            [
                [True, False, True, False],
                [False, True, False, True],
                [True, True, False, False],
                [False, False, True, True],
                [True, False, False, True],
                [False, True, True, False],
                [True, True, True, False],
                [False, False, False, True],
            ]
        )
        self.labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        self.num_features = 4
        self.score_fn = accuracy

    def test_benchmarker_validation(self):
        with self.subTest("score_fn must be callable"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn="not callable",
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                )

        with self.subTest("num_epochs must be int"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5.0,
                    data=self.data,
                    labels=self.labels,
                )

        with self.subTest("num_epochs must be positive"):
            with self.assertRaises(ValueError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=0,
                    data=self.data,
                    labels=self.labels,
                )

        with self.subTest("data must be ndarray"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data="not array",
                    labels=self.labels,
                )

        with self.subTest("labels must be ndarray"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels="not array",
                )

        with self.subTest("labels length must match data rows"):
            with self.assertRaises(ValueError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=np.array([1, 0]),
                )

        with self.subTest("num_runs must be positive"):
            with self.assertRaises(ValueError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    num_runs=0,
                )

        with self.subTest("test_size must be in (0, 1)"):
            with self.assertRaises(ValueError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    test_size=0.0,
                )
            with self.assertRaises(ValueError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    test_size=1.0,
                )
                with self.assertRaises(ValueError):
                    GPBenchmarker(
                        score_fn=self.score_fn,
                        num_epochs=5,
                        data=self.data,
                        labels=self.labels,
                        test_size=-0.1,
                    )

        with self.subTest("n_folds must be at least 2"):
            with self.assertRaises(ValueError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    n_folds=1,
                )

        with self.subTest("val_every must be positive"):
            with self.assertRaises(ValueError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    val_every=0,
                )

        with self.subTest("val_score_fn must be callable if provided"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    val_score_fn="not callable",
                )

        with self.subTest("check_valid must be callable if provided"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    check_valid="not callable",
                )

        with self.subTest("data must be 2D"):
            with self.assertRaises(ValueError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=np.array([1, 2, 3]),  # 1D
                    labels=self.labels,
                )

        with self.subTest("labels must be 1D"):
            with self.assertRaises(ValueError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=np.array([[1, 0], [1, 0]]),  # 2D
                )

        with self.subTest("regeneration_patience must be positive"):
            with self.assertRaises(ValueError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    regeneration_patience=0,
                )

        with self.subTest("num_runs must be int"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    num_runs=2.0,
                )

        with self.subTest("n_folds must be int"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    n_folds=2.0,
                )

        with self.subTest("n_jobs must be int"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    n_jobs=2.0,
                )

        with self.subTest("base_seed must be int"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    base_seed=42.0,
                )

        with self.subTest("progress_bar must be bool"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    progress_bar="yes",
                )

        with self.subTest(
            "population_generator must be PopulationGenerator if provided"
        ):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    population_generator="not a generator",
                )

        with self.subTest("mutation_executor must be MutationExecutor if provided"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    mutation_executor="not an executor",
                )

        with self.subTest("crossover_executor must be CrossoverExecutor if provided"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    crossover_executor="not an executor",
                )

        with self.subTest("selection must be BaseSelection if provided"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    selection="not a selection",
                )

    def test_benchmarker_init(self):
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=5,
            data=self.data,
            labels=self.labels,
            num_runs=2,
            n_folds=2,
            progress_bar=False,
            optimize_scorer=False,
        )
        self.assertEqual(benchmarker.num_runs, 2)
        self.assertEqual(benchmarker.n_folds, 2)
        self.assertEqual(benchmarker.num_epochs, 5)
        self.assertEqual(benchmarker.test_size, 0.2)
        self.assertIsNone(benchmarker._run_metrics)

    def test_benchmarker_defaults(self):
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=5,
            data=self.data,
            labels=self.labels,
            num_runs=2,
            progress_bar=False,
            optimize_scorer=False,
        )
        self.assertEqual(benchmarker.num_runs, 2)
        self.assertEqual(benchmarker.n_folds, 5)
        self.assertEqual(benchmarker.test_size, 0.2)
        self.assertTrue(benchmarker.show_run_progress)
        self.assertTrue(benchmarker.show_fold_progress)
        self.assertTrue(benchmarker.show_epoch_progress)

    def test_fit_returns_benchmark_metrics(self):
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=3,
            data=self.data,
            labels=self.labels,
            num_runs=2,
            n_folds=2,
            n_jobs=1,
            progress_bar=False,
            optimize_scorer=False,
        )
        metrics = benchmarker.fit()

        self.assertIsInstance(metrics, dict)
        self.assertIn("run_metrics", metrics)
        self.assertIn("mean_test_score", metrics)
        self.assertIn("std_test_score", metrics)
        self.assertIn("mean_best_val_score", metrics)
        self.assertIn("std_best_val_score", metrics)
        self.assertIn("all_test_scores", metrics)
        self.assertEqual(len(metrics["run_metrics"]), 2)
        self.assertEqual(len(metrics["all_test_scores"]), 2)

    def test_fit_run_metrics_structure(self):
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=3,
            data=self.data,
            labels=self.labels,
            num_runs=1,
            n_folds=2,
            n_jobs=1,
            progress_bar=False,
            optimize_scorer=False,
        )
        metrics = benchmarker.fit()

        run_metrics = metrics["run_metrics"][0]
        self.assertIn("run_id", run_metrics)
        self.assertIn("seed", run_metrics)
        self.assertIn("fold_train_scores", run_metrics)
        self.assertIn("fold_val_scores", run_metrics)
        self.assertIn("best_fold_idx", run_metrics)
        self.assertIn("best_fold_val_score", run_metrics)
        self.assertIn("test_score", run_metrics)
        self.assertIn("best_rule", run_metrics)
        self.assertEqual(run_metrics["run_id"], 0)
        self.assertEqual(len(run_metrics["fold_train_scores"]), 2)
        self.assertEqual(len(run_metrics["fold_val_scores"]), 2)
        self.assertIsInstance(run_metrics["best_rule"], Rule)

    def test_fit_aggregation(self):
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=3,
            data=self.data,
            labels=self.labels,
            num_runs=3,
            n_folds=2,
            n_jobs=1,
            progress_bar=False,
            optimize_scorer=False,
        )
        metrics = benchmarker.fit()

        all_test = metrics["all_test_scores"]
        self.assertAlmostEqual(metrics["mean_test_score"], np.mean(all_test))
        self.assertAlmostEqual(metrics["std_test_score"], np.std(all_test))

        all_val = [m["best_fold_val_score"] for m in metrics["run_metrics"]]
        self.assertAlmostEqual(metrics["mean_best_val_score"], np.mean(all_val))
        self.assertAlmostEqual(metrics["std_best_val_score"], np.std(all_val))

    def test_fit_sequential_n_jobs_one(self):
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=2,
            n_folds=2,
            n_jobs=1,
            progress_bar=False,
            optimize_scorer=False,
        )
        metrics = benchmarker.fit()
        self.assertEqual(len(metrics["run_metrics"]), 2)
        self.assertIsNotNone(benchmarker._run_metrics)

    def test_fit_parallel_n_jobs_two(self):
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=2,
            n_folds=2,
            n_jobs=2,
            progress_bar=False,
            optimize_scorer=False,
        )
        metrics = benchmarker.fit()
        self.assertEqual(len(metrics["run_metrics"]), 2)
        self.assertIsNotNone(benchmarker._run_metrics)

    def test_reproducibility_with_seed(self):
        benchmarker1 = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=2,
            n_folds=2,
            n_jobs=1,
            base_seed=12345,
            progress_bar=False,
            optimize_scorer=False,
        )
        metrics1 = benchmarker1.fit()

        benchmarker2 = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=2,
            n_folds=2,
            n_jobs=1,
            base_seed=12345,
            progress_bar=False,
            optimize_scorer=False,
        )
        metrics2 = benchmarker2.fit()

        self.assertEqual(
            [m["seed"] for m in metrics1["run_metrics"]],
            [m["seed"] for m in metrics2["run_metrics"]],
        )
        self.assertEqual(metrics1["mean_test_score"], metrics2["mean_test_score"])

    def test_custom_population_generator(self):
        generator = PopulationGenerator(
            strategies=[RandomStrategy(num_literals=self.num_features)],
            population_size=10,
        )
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=1,
            n_folds=2,
            n_jobs=1,
            population_generator=generator,
            progress_bar=False,
            optimize_scorer=False,
        )
        metrics = benchmarker.fit()
        self.assertEqual(len(metrics["run_metrics"]), 1)

    def test_custom_mutation_executor(self):
        mutation_executor = MutationExecutor(
            literal_mutations=create_standard_literal_mutations(self.num_features),
            operator_mutations=create_standard_operator_mutations(self.num_features),
            mutation_p=0.1,
        )
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=1,
            n_folds=2,
            n_jobs=1,
            mutation_executor=mutation_executor,
            progress_bar=False,
            optimize_scorer=False,
        )
        metrics = benchmarker.fit()
        self.assertEqual(len(metrics["run_metrics"]), 1)

    def test_custom_crossover_executor(self):
        crossover_executor = CrossoverExecutor(crossover_p=0.5)
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=1,
            n_folds=2,
            n_jobs=1,
            crossover_executor=crossover_executor,
            progress_bar=False,
            optimize_scorer=False,
        )
        metrics = benchmarker.fit()
        self.assertEqual(len(metrics["run_metrics"]), 1)

    def test_custom_selection(self):
        selection = RouletteSelection()
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=1,
            n_folds=2,
            n_jobs=1,
            selection=selection,
            progress_bar=False,
            optimize_scorer=False,
        )
        metrics = benchmarker.fit()
        self.assertEqual(len(metrics["run_metrics"]), 1)

    def test_regeneration(self):
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=1,
            n_folds=2,
            n_jobs=1,
            regeneration=True,
            regeneration_patience=50,
            progress_bar=False,
            optimize_scorer=False,
        )
        metrics = benchmarker.fit()
        self.assertEqual(len(metrics["run_metrics"]), 1)

    def test_val_score_fn(self):
        def custom_val_score(predictions, labels):
            return np.sum(predictions & labels)

        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=1,
            n_folds=2,
            n_jobs=1,
            val_score_fn=custom_val_score,
            progress_bar=False,
            optimize_scorer=False,
        )
        metrics = benchmarker.fit()
        self.assertEqual(len(metrics["run_metrics"]), 1)

    def test_progress_bar_disabled(self):
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=2,
            n_folds=2,
            n_jobs=1,
            progress_bar=False,
            optimize_scorer=False,
        )
        self.assertFalse(benchmarker.progress_bar)
        metrics = benchmarker.fit()
        self.assertEqual(len(metrics["run_metrics"]), 2)

    def test_optimize_scorer_validation(self):
        with self.subTest("optimize_scorer must be bool"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    score_fn=self.score_fn,
                    num_epochs=5,
                    data=self.data,
                    labels=self.labels,
                    optimize_scorer="yes",
                )

    def test_optimize_scorer_true_with_sample_weight(self):
        """Test that optimize_scorer=True works with a scorer supporting sample_weight."""
        benchmarker = GPBenchmarker(
            score_fn=accuracy_with_sample_weight,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=1,
            n_folds=2,
            n_jobs=1,
            optimize_scorer=True,
            progress_bar=False,
        )
        metrics = benchmarker.fit()
        self.assertEqual(len(metrics["run_metrics"]), 1)
        self.assertIn("test_score", metrics["run_metrics"][0])

    def test_optimize_scorer_false(self):
        """Test that optimize_scorer=False works without sample_weight support."""
        benchmarker = GPBenchmarker(
            score_fn=self.score_fn,  # accuracy without sample_weight
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=1,
            n_folds=2,
            n_jobs=1,
            optimize_scorer=False,
            progress_bar=False,
        )
        metrics = benchmarker.fit()
        self.assertEqual(len(metrics["run_metrics"]), 1)
        self.assertIn("test_score", metrics["run_metrics"][0])

    def test_optimize_scorer_default_is_true(self):
        """Test that optimize_scorer defaults to True."""
        benchmarker = GPBenchmarker(
            score_fn=accuracy_with_sample_weight,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=1,
            n_folds=2,
            n_jobs=1,
            progress_bar=False,
        )
        self.assertTrue(benchmarker.optimize_scorer)

    def test_optimize_scorer_with_different_val_score_fn(self):
        """Test optimize_scorer with separate train and val score functions."""
        benchmarker = GPBenchmarker(
            score_fn=accuracy_with_sample_weight,
            val_score_fn=accuracy_with_sample_weight,
            num_epochs=2,
            data=self.data,
            labels=self.labels,
            num_runs=1,
            n_folds=2,
            n_jobs=1,
            optimize_scorer=True,
            progress_bar=False,
        )
        metrics = benchmarker.fit()
        self.assertEqual(len(metrics["run_metrics"]), 1)

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.benchmarkers.gp_benchmarker, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.benchmarkers.config, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.benchmarkers.results, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.benchmarkers.runner, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
