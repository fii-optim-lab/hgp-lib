import doctest
import unittest
import random

import numpy as np

import hgp_lib
from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
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

    def _make_gp_config(self, **kwargs):
        """Helper to create BooleanGPConfig with test defaults."""
        defaults = dict(
            score_fn=self.score_fn,
            optimize_scorer=False,
        )
        defaults.update(kwargs)
        return BooleanGPConfig(**defaults)

    def _make_trainer_config(self, gp_config=None, **kwargs):
        """Helper to create TrainerConfig with test defaults."""
        if gp_config is None:
            gp_config = self._make_gp_config()
        defaults = dict(
            gp_config=gp_config,
            num_epochs=5,
            progress_bar=False,
        )
        defaults.update(kwargs)
        return TrainerConfig(**defaults)

    def _make_config(self, trainer_config=None, **kwargs):
        """Helper to create BenchmarkerConfig with test defaults."""
        if trainer_config is None:
            trainer_config = self._make_trainer_config()
        defaults = dict(
            data=self.data,
            labels=self.labels,
            trainer_config=trainer_config,
            num_runs=2,
            n_folds=2,
        )
        defaults.update(kwargs)
        return BenchmarkerConfig(**defaults)

    def test_benchmarker_validation(self):
        with self.subTest("score_fn must be callable"):
            with self.assertRaises(TypeError):
                gp_config = self._make_gp_config(score_fn="not callable")
                trainer_config = self._make_trainer_config(gp_config=gp_config)
                config = self._make_config(trainer_config=trainer_config)
                GPBenchmarker(config)

        with self.subTest("num_epochs must be int"):
            with self.assertRaises(TypeError):
                trainer_config = self._make_trainer_config(num_epochs=5.0)
                config = self._make_config(trainer_config=trainer_config)
                GPBenchmarker(config)

        with self.subTest("num_epochs must be positive"):
            with self.assertRaises(ValueError):
                trainer_config = self._make_trainer_config(num_epochs=0)
                config = self._make_config(trainer_config=trainer_config)
                GPBenchmarker(config)

        with self.subTest("data must be ndarray"):
            with self.assertRaises(TypeError):
                config = self._make_config(data="not array")
                GPBenchmarker(config)

        with self.subTest("labels must be ndarray"):
            with self.assertRaises(TypeError):
                config = self._make_config(labels="not array")
                GPBenchmarker(config)

        with self.subTest("labels length must match data rows"):
            with self.assertRaises(ValueError):
                config = self._make_config(labels=np.array([1, 0]))
                GPBenchmarker(config)

        with self.subTest("num_runs must be positive"):
            with self.assertRaises(ValueError):
                config = self._make_config(num_runs=0)
                GPBenchmarker(config)

        with self.subTest("test_size must be in (0, 1)"):
            with self.assertRaises(ValueError):
                config = self._make_config(test_size=0.0)
                GPBenchmarker(config)
            with self.assertRaises(ValueError):
                config = self._make_config(test_size=1.0)
                GPBenchmarker(config)

        with self.subTest("n_folds must be at least 2"):
            with self.assertRaises(ValueError):
                config = self._make_config(n_folds=1)
                GPBenchmarker(config)

        with self.subTest("check_valid must be callable if provided"):
            with self.assertRaises(TypeError):
                gp_config = self._make_gp_config(check_valid="not callable")
                trainer_config = self._make_trainer_config(gp_config=gp_config)
                config = self._make_config(trainer_config=trainer_config)
                GPBenchmarker(config)

        with self.subTest("data must be 2D"):
            with self.assertRaises(ValueError):
                config = self._make_config(data=np.array([1, 2, 3]))
                GPBenchmarker(config)

        with self.subTest("labels must be 1D"):
            with self.assertRaises(ValueError):
                config = self._make_config(labels=np.array([[1, 0], [1, 0]]))
                GPBenchmarker(config)

        with self.subTest(
            "population_generator must be PopulationGenerator if provided"
        ):
            with self.assertRaises(TypeError):
                gp_config = self._make_gp_config(population_generator="not a generator")
                trainer_config = self._make_trainer_config(gp_config=gp_config)
                config = self._make_config(trainer_config=trainer_config)
                GPBenchmarker(config)

        with self.subTest("mutation_executor must be MutationExecutor if provided"):
            with self.assertRaises(TypeError):
                gp_config = self._make_gp_config(mutation_executor="not an executor")
                trainer_config = self._make_trainer_config(gp_config=gp_config)
                config = self._make_config(trainer_config=trainer_config)
                GPBenchmarker(config)

        with self.subTest("crossover_executor must be CrossoverExecutor if provided"):
            with self.assertRaises(TypeError):
                gp_config = self._make_gp_config(crossover_executor="not an executor")
                trainer_config = self._make_trainer_config(gp_config=gp_config)
                config = self._make_config(trainer_config=trainer_config)
                GPBenchmarker(config)

        with self.subTest("selection must be BaseSelection if provided"):
            with self.assertRaises(TypeError):
                gp_config = self._make_gp_config(selection="not a selection")
                trainer_config = self._make_trainer_config(gp_config=gp_config)
                config = self._make_config(trainer_config=trainer_config)
                GPBenchmarker(config)

    def test_benchmarker_init(self):
        config = self._make_config()
        benchmarker = GPBenchmarker(config)
        self.assertEqual(benchmarker.config.num_runs, 2)
        self.assertEqual(benchmarker.config.n_folds, 2)
        self.assertEqual(benchmarker.config.trainer_config.num_epochs, 5)
        self.assertEqual(benchmarker.config.test_size, 0.2)
        self.assertIsNone(benchmarker._run_metrics)

    def test_benchmarker_defaults(self):
        config = self._make_config()
        benchmarker = GPBenchmarker(config)
        self.assertEqual(benchmarker.config.num_runs, 2)
        self.assertEqual(benchmarker.config.n_folds, 2)
        self.assertEqual(benchmarker.config.test_size, 0.2)
        self.assertTrue(benchmarker.config.show_run_progress)
        self.assertTrue(benchmarker.config.show_fold_progress)
        self.assertTrue(benchmarker.config.show_epoch_progress)

    def test_fit_returns_benchmark_result(self):
        trainer_config = self._make_trainer_config(num_epochs=3)
        config = self._make_config(trainer_config=trainer_config, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()

        self.assertIsNotNone(result.run_metrics)
        self.assertIsNotNone(result.mean_test_score)
        self.assertIsNotNone(result.std_test_score)
        self.assertIsNotNone(result.mean_best_val_score)
        self.assertIsNotNone(result.std_best_val_score)
        self.assertIsNotNone(result.all_test_scores)
        self.assertEqual(len(result.run_metrics), 2)
        self.assertEqual(len(result.all_test_scores), 2)

    def test_fit_run_metrics_structure(self):
        trainer_config = self._make_trainer_config(num_epochs=3)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()

        run_metrics = result.run_metrics[0]
        self.assertIsNotNone(run_metrics.run_id)
        self.assertIsNotNone(run_metrics.seed)
        self.assertIsNotNone(run_metrics.fold_train_scores)
        self.assertIsNotNone(run_metrics.fold_val_scores)
        self.assertIsNotNone(run_metrics.best_fold_idx)
        self.assertIsNotNone(run_metrics.best_fold_val_score)
        self.assertIsNotNone(run_metrics.test_score)
        self.assertIsNotNone(run_metrics.best_rule)
        self.assertEqual(run_metrics.run_id, 0)
        self.assertEqual(len(run_metrics.fold_train_scores), 2)
        self.assertEqual(len(run_metrics.fold_val_scores), 2)
        self.assertIsInstance(run_metrics.best_rule, Rule)

    def test_fit_aggregation(self):
        trainer_config = self._make_trainer_config(num_epochs=3)
        config = self._make_config(trainer_config=trainer_config, num_runs=3, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()

        all_test = result.all_test_scores
        self.assertAlmostEqual(result.mean_test_score, np.mean(all_test))
        self.assertAlmostEqual(result.std_test_score, np.std(all_test))

        all_val = [m.best_fold_val_score for m in result.run_metrics]
        self.assertAlmostEqual(result.mean_best_val_score, np.mean(all_val))
        self.assertAlmostEqual(result.std_best_val_score, np.std(all_val))

    def test_fit_sequential_n_jobs_one(self):
        trainer_config = self._make_trainer_config(num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.run_metrics), 2)
        self.assertIsNotNone(benchmarker._run_metrics)

    def test_fit_parallel_n_jobs_two(self):
        trainer_config = self._make_trainer_config(num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, n_jobs=2)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.run_metrics), 2)
        self.assertIsNotNone(benchmarker._run_metrics)

    def test_reproducibility_with_seed(self):
        trainer_config = self._make_trainer_config(num_epochs=2)
        config1 = self._make_config(trainer_config=trainer_config, n_jobs=1, base_seed=12345)
        benchmarker1 = GPBenchmarker(config1)
        result1 = benchmarker1.fit()

        config2 = self._make_config(trainer_config=trainer_config, n_jobs=1, base_seed=12345)
        benchmarker2 = GPBenchmarker(config2)
        result2 = benchmarker2.fit()

        self.assertEqual(
            [m.seed for m in result1.run_metrics],
            [m.seed for m in result2.run_metrics],
        )
        self.assertEqual(result1.mean_test_score, result2.mean_test_score)

    def test_custom_population_generator(self):
        generator = PopulationGenerator(
            strategies=[RandomStrategy(num_literals=self.num_features)],
            population_size=10,
        )
        gp_config = self._make_gp_config(population_generator=generator)
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.run_metrics), 1)

    def test_custom_mutation_executor(self):
        mutation_executor = MutationExecutor(
            literal_mutations=create_standard_literal_mutations(self.num_features),
            operator_mutations=create_standard_operator_mutations(self.num_features),
            mutation_p=0.1,
        )
        gp_config = self._make_gp_config(mutation_executor=mutation_executor)
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.run_metrics), 1)

    def test_custom_crossover_executor(self):
        crossover_executor = CrossoverExecutor(crossover_p=0.5)
        gp_config = self._make_gp_config(crossover_executor=crossover_executor)
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.run_metrics), 1)

    def test_custom_selection(self):
        selection = RouletteSelection()
        gp_config = self._make_gp_config(selection=selection)
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.run_metrics), 1)

    def test_regeneration(self):
        gp_config = self._make_gp_config(regeneration=True, regeneration_patience=50)
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.run_metrics), 1)

    def test_val_score_fn(self):
        def custom_val_score(predictions, labels):
            return np.sum(predictions & labels)

        trainer_config = self._make_trainer_config(
            num_epochs=2, val_score_fn=custom_val_score
        )
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.run_metrics), 1)

    def test_progress_bar_disabled(self):
        trainer_config = self._make_trainer_config(num_epochs=2, progress_bar=False)
        config = self._make_config(trainer_config=trainer_config)
        benchmarker = GPBenchmarker(config)
        self.assertFalse(benchmarker.config.trainer_config.progress_bar)
        result = benchmarker.fit()
        self.assertEqual(len(result.run_metrics), 2)

    def test_optimize_scorer_true_with_sample_weight(self):
        """Test that optimize_scorer=True works with a scorer supporting sample_weight."""
        gp_config = self._make_gp_config(
            score_fn=accuracy_with_sample_weight, optimize_scorer=True
        )
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.run_metrics), 1)
        self.assertIsNotNone(result.run_metrics[0].test_score)

    def test_optimize_scorer_false(self):
        """Test that optimize_scorer=False works without sample_weight support."""
        gp_config = self._make_gp_config(optimize_scorer=False)
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.run_metrics), 1)
        self.assertIsNotNone(result.run_metrics[0].test_score)

    def test_optimize_scorer_default_is_false_in_gp_config(self):
        """Test that optimize_scorer defaults to False in BooleanGPConfig."""
        gp_config = BooleanGPConfig(score_fn=accuracy_with_sample_weight)
        self.assertFalse(gp_config.optimize_scorer)

    def test_optimize_scorer_with_different_val_score_fn(self):
        """Test optimize_scorer with separate train and val score functions."""
        gp_config = self._make_gp_config(
            score_fn=accuracy_with_sample_weight, optimize_scorer=True
        )
        trainer_config = self._make_trainer_config(
            gp_config=gp_config, num_epochs=2, val_score_fn=accuracy_with_sample_weight
        )
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.run_metrics), 1)

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.benchmarkers.gp_benchmarker, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.benchmarkers.results, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.benchmarkers.runner, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
