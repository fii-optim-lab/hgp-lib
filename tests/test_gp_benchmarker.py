import doctest
import unittest
import random

import numpy as np
import pandas as pd

import hgp_lib
from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.metrics import ExperimentResult, RunResult
from hgp_lib.mutations import MutationExecutorFactory
from hgp_lib.populations import PopulationGeneratorFactory
from hgp_lib.preprocessing import StandardBinarizer
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

        self.data = pd.DataFrame(
            np.array(
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
            ),
            columns=["0", "1", "2", "3"],
        )
        self.labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        self.num_features = 4
        self.score_fn = accuracy

    def _make_gp_config(self, **kwargs):
        defaults = dict(
            score_fn=self.score_fn,
            optimize_scorer=False,
        )
        defaults.update(kwargs)
        return BooleanGPConfig(**defaults)

    def _make_trainer_config(self, gp_config=None, **kwargs):
        if gp_config is None:
            gp_config = self._make_gp_config()
        defaults = dict(
            gp_config=gp_config,
            num_epochs=5,
            val_every=1,
            progress_bar=False,
        )
        defaults.update(kwargs)
        return TrainerConfig(**defaults)

    def _make_config(self, trainer_config=None, **kwargs):
        if trainer_config is None:
            trainer_config = self._make_trainer_config()
        defaults = dict(
            data=self.data,
            labels=self.labels,
            trainer_config=trainer_config,
            num_runs=2,
            n_folds=2,
            n_jobs=1,
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

        with self.subTest("data must be a DataFrame"):
            with self.assertRaises(TypeError):
                config = self._make_config(data="not a dataframe")
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
                config = self._make_config(data=pd.DataFrame(np.array([1, 2, 3])))
                GPBenchmarker(config)

        with self.subTest("labels must be 1D"):
            with self.assertRaises(ValueError):
                config = self._make_config(labels=np.array([[1, 0], [1, 0]]))
                GPBenchmarker(config)

        with self.subTest("population_factory must be PopulationGeneratorFactory"):
            with self.assertRaises(TypeError):
                gp_config = self._make_gp_config(population_factory="not a factory")
                trainer_config = self._make_trainer_config(gp_config=gp_config)
                config = self._make_config(trainer_config=trainer_config)
                GPBenchmarker(config)

        with self.subTest("mutation_factory must be MutationExecutorFactory"):
            with self.assertRaises(TypeError):
                gp_config = self._make_gp_config(mutation_factory="not a factory")
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
        self.assertIsNone(benchmarker._run_results)

    def test_benchmarker_defaults(self):
        config = self._make_config()
        benchmarker = GPBenchmarker(config)
        self.assertEqual(benchmarker.config.num_runs, 2)
        self.assertEqual(benchmarker.config.n_folds, 2)
        self.assertEqual(benchmarker.config.test_size, 0.2)
        self.assertTrue(benchmarker.config.show_run_progress)
        self.assertTrue(benchmarker.config.show_fold_progress)
        self.assertTrue(benchmarker.config.show_epoch_progress)

    def test_fit_returns_experiment_result(self):
        trainer_config = self._make_trainer_config(num_epochs=3)
        config = self._make_config(trainer_config=trainer_config, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()

        self.assertIsInstance(result, ExperimentResult)
        self.assertEqual(len(result.runs), 2)

        test_scores = result.test_scores
        self.assertIsNotNone(test_scores)
        self.assertEqual(len(test_scores), 2)

    def test_fit_run_result_structure(self):
        trainer_config = self._make_trainer_config(num_epochs=3)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()

        run_result = result.runs[0]
        self.assertIsInstance(run_result, RunResult)
        self.assertIsNotNone(run_result.run_id)
        self.assertIsNotNone(run_result.seed)
        self.assertIsNotNone(run_result.folds)
        self.assertIsNotNone(run_result.test_score)
        self.assertIsNotNone(run_result.feature_names)

        self.assertEqual(run_result.run_id, 0)
        self.assertEqual(len(run_result.folds), 2)

        self.assertIsInstance(run_result.feature_names, dict)
        self.assertTrue(len(run_result.feature_names) > 0)
        for idx, name in run_result.feature_names.items():
            self.assertIsInstance(idx, int)
            self.assertIsInstance(name, str)

    def test_fit_aggregation(self):
        trainer_config = self._make_trainer_config(num_epochs=3)
        config = self._make_config(trainer_config=trainer_config, num_runs=3, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()

        all_test = result.test_scores
        mean_test = float(np.mean(all_test))
        std_test = float(np.std(all_test))

        self.assertEqual(len(all_test), 3)
        self.assertIsInstance(mean_test, float)
        self.assertIsInstance(std_test, float)

    def test_fit_sequential_n_jobs_one(self):
        trainer_config = self._make_trainer_config(num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.runs), 2)
        self.assertIsNotNone(benchmarker._run_results)

    def test_fit_parallel_n_jobs_two(self):
        trainer_config = self._make_trainer_config(num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, n_jobs=2)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.runs), 2)
        self.assertIsNotNone(benchmarker._run_results)

    def test_reproducibility_with_seed(self):
        trainer_config = self._make_trainer_config(num_epochs=2)
        config1 = self._make_config(
            trainer_config=trainer_config, n_jobs=1, base_seed=12345
        )
        benchmarker1 = GPBenchmarker(config1)
        result1 = benchmarker1.fit()

        config2 = self._make_config(
            trainer_config=trainer_config, n_jobs=1, base_seed=12345
        )
        benchmarker2 = GPBenchmarker(config2)
        result2 = benchmarker2.fit()

        self.assertEqual(
            [r.seed for r in result1.runs],
            [r.seed for r in result2.runs],
        )
        test_scores_1 = result1.test_scores
        test_scores_2 = result2.test_scores
        self.assertEqual(float(np.mean(test_scores_1)), float(np.mean(test_scores_2)))

    def test_custom_population_factory(self):
        factory = PopulationGeneratorFactory(population_size=10)
        gp_config = self._make_gp_config(population_factory=factory)
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.runs), 1)

    def test_custom_mutation_factory(self):
        factory = MutationExecutorFactory(mutation_p=0.1)
        gp_config = self._make_gp_config(mutation_factory=factory)
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.runs), 1)

    def test_custom_crossover_executor(self):
        crossover_executor = CrossoverExecutor(crossover_p=0.5)
        gp_config = self._make_gp_config(crossover_executor=crossover_executor)
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.runs), 1)

    def test_custom_selection(self):
        selection = RouletteSelection()
        gp_config = self._make_gp_config(selection=selection)
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.runs), 1)

    def test_regeneration(self):
        gp_config = self._make_gp_config(regeneration=True, regeneration_patience=50)
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.runs), 1)

    def test_progress_bar_disabled(self):
        trainer_config = self._make_trainer_config(num_epochs=2, progress_bar=False)
        config = self._make_config(trainer_config=trainer_config)
        benchmarker = GPBenchmarker(config)
        self.assertFalse(benchmarker.config.trainer_config.progress_bar)
        result = benchmarker.fit()
        self.assertEqual(len(result.runs), 2)

    def test_optimize_scorer_true_with_sample_weight(self):
        gp_config = self._make_gp_config(
            score_fn=accuracy_with_sample_weight, optimize_scorer=True
        )
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.runs), 1)
        self.assertIsNotNone(result.runs[0].test_score)

    def test_optimize_scorer_false(self):
        gp_config = self._make_gp_config(optimize_scorer=False)
        trainer_config = self._make_trainer_config(gp_config=gp_config, num_epochs=2)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()
        self.assertEqual(len(result.runs), 1)
        self.assertIsNotNone(result.runs[0].test_score)

    def test_optimize_scorer_default_is_true_in_gp_config(self):
        gp_config = BooleanGPConfig(score_fn=accuracy_with_sample_weight)
        self.assertTrue(gp_config.optimize_scorer)

    def test_default_binarizer_is_set(self):
        config = self._make_config()
        benchmarker = GPBenchmarker(config)
        self.assertIsInstance(benchmarker.config.binarizer, StandardBinarizer)

    def test_custom_binarizer(self):
        binarizer = StandardBinarizer(num_bins=3)
        config = self._make_config(binarizer=binarizer)
        benchmarker = GPBenchmarker(config)
        self.assertIs(benchmarker.config.binarizer, binarizer)
        result = benchmarker.fit()
        self.assertEqual(len(result.runs), 2)

    def test_fitted_binarizer_rejected(self):
        binarizer = StandardBinarizer(num_bins=2)
        binarizer.fit_transform(self.data)
        self.assertTrue(binarizer._is_fitted)
        with self.assertRaises(ValueError):
            config = self._make_config(binarizer=binarizer)
            GPBenchmarker(config)

    def test_data_must_be_dataframe(self):
        with self.assertRaises(TypeError):
            config = self._make_config(data=self.data.to_numpy())
            GPBenchmarker(config)

    def test_feature_names_enable_readable_rules(self):
        trainer_config = self._make_trainer_config(num_epochs=3)
        config = self._make_config(trainer_config=trainer_config, num_runs=1, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()

        best_run = result.best_run
        best_rule = result.best_rule
        feature_names = best_run.feature_names

        readable = best_rule.to_str(feature_names)
        self.assertIsInstance(readable, str)
        self.assertTrue(len(readable) > 0)

    def test_best_rule(self):
        trainer_config = self._make_trainer_config(num_epochs=3)
        config = self._make_config(trainer_config=trainer_config, num_runs=2, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()

        best_rule = result.best_rule
        self.assertIsInstance(best_rule, Rule)

    def test_best_run(self):
        trainer_config = self._make_trainer_config(num_epochs=3)
        config = self._make_config(trainer_config=trainer_config, num_runs=2, n_jobs=1)
        benchmarker = GPBenchmarker(config)
        result = benchmarker.fit()

        best_run = result.best_run
        self.assertIsInstance(best_run, RunResult)

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.benchmarkers.gp_benchmarker, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.benchmarkers.runner, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
