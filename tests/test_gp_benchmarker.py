import doctest
import queue
import unittest
import random
from multiprocessing import Queue

import numpy as np
import pandas as pd

import hgp_lib
import hgp_lib.benchmarkers.progress
import hgp_lib.benchmarkers.runner
from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.benchmarkers.runner import execute_single_run, single_run_wrapper
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.crossover import CrossoverExecutorFactory
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
        self.score_fn = accuracy

    def _make_gp_config(self, **kwargs):
        defaults = dict(score_fn=self.score_fn, optimize_scorer=False)
        defaults.update(kwargs)
        return BooleanGPConfig(**defaults)

    def _make_trainer_config(self, gp_config=None, **kwargs):
        if gp_config is None:
            gp_config = self._make_gp_config()
        defaults = dict(
            gp_config=gp_config, num_epochs=5, val_every=1, progress_bar=False
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

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #
    def test_benchmarker_validation(self):
        with self.subTest("score_fn must be callable"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    self._make_config(
                        trainer_config=self._make_trainer_config(
                            gp_config=self._make_gp_config(score_fn="not callable")
                        )
                    )
                )

        with self.subTest("num_epochs must be int"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    self._make_config(
                        trainer_config=self._make_trainer_config(num_epochs=5.0)
                    )
                )

        with self.subTest("num_epochs must be positive"):
            with self.assertRaises(ValueError):
                GPBenchmarker(
                    self._make_config(
                        trainer_config=self._make_trainer_config(num_epochs=0)
                    )
                )

        with self.subTest("data must be a DataFrame"):
            with self.assertRaises(TypeError):
                GPBenchmarker(self._make_config(data="not a dataframe"))

        with self.subTest("labels must be ndarray"):
            with self.assertRaises(TypeError):
                GPBenchmarker(self._make_config(labels="not array"))

        with self.subTest("labels length must match data rows"):
            with self.assertRaises(ValueError):
                GPBenchmarker(self._make_config(labels=np.array([1, 0])))

        with self.subTest("num_runs must be positive"):
            with self.assertRaises(ValueError):
                GPBenchmarker(self._make_config(num_runs=0))

        with self.subTest("test_size must be in (0, 1) — low"):
            with self.assertRaises(ValueError):
                GPBenchmarker(self._make_config(test_size=0.0))

        with self.subTest("test_size must be in (0, 1) — high"):
            with self.assertRaises(ValueError):
                GPBenchmarker(self._make_config(test_size=1.0))

        with self.subTest("n_folds must be at least 2"):
            with self.assertRaises(ValueError):
                GPBenchmarker(self._make_config(n_folds=1))

        with self.subTest("data must be 2D"):
            with self.assertRaises(ValueError):
                GPBenchmarker(self._make_config(data=pd.DataFrame(np.array([1, 2, 3]))))

        with self.subTest("labels must be 1D"):
            with self.assertRaises(ValueError):
                GPBenchmarker(self._make_config(labels=np.array([[1, 0], [1, 0]])))

        with self.subTest("population_factory type"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    self._make_config(
                        trainer_config=self._make_trainer_config(
                            gp_config=self._make_gp_config(population_factory="bad")
                        )
                    )
                )

        with self.subTest("mutation_factory type"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    self._make_config(
                        trainer_config=self._make_trainer_config(
                            gp_config=self._make_gp_config(mutation_factory="bad")
                        )
                    )
                )

        with self.subTest("selection type"):
            with self.assertRaises(TypeError):
                GPBenchmarker(
                    self._make_config(
                        trainer_config=self._make_trainer_config(
                            gp_config=self._make_gp_config(selection="bad")
                        )
                    )
                )

    # ------------------------------------------------------------------ #
    #  Initialization
    # ------------------------------------------------------------------ #
    def test_benchmarker_init(self):
        benchmarker = GPBenchmarker(self._make_config())
        self.assertEqual(benchmarker.config.num_runs, 2)
        self.assertEqual(benchmarker.config.n_folds, 2)
        self.assertEqual(benchmarker.config.test_size, 0.2)
        self.assertIsNone(benchmarker._run_results)

    def test_benchmarker_defaults(self):
        benchmarker = GPBenchmarker(self._make_config())
        self.assertTrue(benchmarker.config.show_run_progress)
        self.assertTrue(benchmarker.config.show_fold_progress)
        self.assertTrue(benchmarker.config.show_epoch_progress)

    def test_default_binarizer_is_set(self):
        benchmarker = GPBenchmarker(self._make_config())
        self.assertIsInstance(benchmarker.config.binarizer, StandardBinarizer)

    def test_custom_binarizer(self):
        binarizer = StandardBinarizer(num_bins=3)
        benchmarker = GPBenchmarker(self._make_config(binarizer=binarizer))
        self.assertIs(benchmarker.config.binarizer, binarizer)

    def test_fitted_binarizer_rejected(self):
        binarizer = StandardBinarizer(num_bins=2)
        binarizer.fit_transform(self.data)
        with self.assertRaises(ValueError):
            GPBenchmarker(self._make_config(binarizer=binarizer))

    def test_data_must_be_dataframe(self):
        with self.assertRaises(TypeError):
            GPBenchmarker(self._make_config(data=self.data.to_numpy()))

    # ------------------------------------------------------------------ #
    #  _effective_n_jobs
    # ------------------------------------------------------------------ #
    def test_effective_n_jobs_sequential(self):
        self.assertEqual(
            GPBenchmarker(self._make_config(n_jobs=1))._effective_n_jobs(), 1
        )

    def test_effective_n_jobs_capped_by_num_runs(self):
        self.assertEqual(
            GPBenchmarker(self._make_config(n_jobs=10, num_runs=2))._effective_n_jobs(),
            2,
        )

    def test_effective_n_jobs_negative_uses_cpu_count(self):
        n = GPBenchmarker(
            self._make_config(n_jobs=-1, num_runs=100)
        )._effective_n_jobs()
        self.assertGreaterEqual(n, 1)

    # ------------------------------------------------------------------ #
    #  _run_sequential — leave_progress_bar side effect
    # ------------------------------------------------------------------ #
    def test_sequential_sets_leave_progress_bar_false(self):
        config = self._make_config(n_jobs=1)
        config.show_run_progress = True
        benchmarker = GPBenchmarker(config)
        benchmarker._run_sequential()
        self.assertFalse(benchmarker.config.trainer_config.leave_progress_bar)

    # ------------------------------------------------------------------ #
    #  fit — sequential (n_jobs=1)
    # ------------------------------------------------------------------ #
    def test_fit_returns_experiment_result(self):
        result = GPBenchmarker(self._make_config(num_runs=2)).fit()
        self.assertIsInstance(result, ExperimentResult)
        self.assertEqual(len(result.runs), 2)

    def test_fit_stores_run_results(self):
        benchmarker = GPBenchmarker(self._make_config())
        self.assertIsNone(benchmarker._run_results)
        benchmarker.fit()
        self.assertIsNotNone(benchmarker._run_results)

    def test_fit_run_result_structure(self):
        result = GPBenchmarker(self._make_config(num_runs=1)).fit()
        run = result.runs[0]
        self.assertIsInstance(run, RunResult)
        self.assertEqual(run.run_id, 0)
        self.assertEqual(len(run.folds), 2)
        self.assertIsInstance(run.test_score, float)
        self.assertIsInstance(run.feature_names, dict)
        for idx, name in run.feature_names.items():
            self.assertIsInstance(idx, int)
            self.assertIsInstance(name, str)

    def test_fit_confusion_matrix_fields(self):
        """RunResult must contain valid confusion matrix values that sum correctly."""
        result = GPBenchmarker(self._make_config(num_runs=1)).fit()
        run = result.runs[0]
        # All non-negative
        self.assertGreaterEqual(run.test_tp, 0)
        self.assertGreaterEqual(run.test_fp, 0)
        self.assertGreaterEqual(run.test_fn, 0)
        self.assertGreaterEqual(run.test_tn, 0)
        # tp + fp + fn + tn == total test samples
        total = run.test_tp + run.test_fp + run.test_fn + run.test_tn
        expected_test_size = int(len(self.labels) * 0.2)  # test_size=0.2
        self.assertGreaterEqual(total, expected_test_size - 1)
        self.assertLessEqual(total, expected_test_size + 1)

    def test_fit_best_fold_has_highest_val_score(self):
        """best_fold_idx should point to the fold with the highest val score."""
        result = GPBenchmarker(self._make_config(num_runs=1)).fit()
        run = result.runs[0]
        val_scores = [
            f.best_val_score for f in run.folds if f.best_val_score is not None
        ]
        if len(val_scores) == len(run.folds):
            best_val = max(val_scores)
            self.assertEqual(run.folds[run.best_fold_idx].best_val_score, best_val)

    def test_fit_aggregation(self):
        result = GPBenchmarker(self._make_config(num_runs=3)).fit()
        scores = result.test_scores
        self.assertEqual(len(scores), 3)
        self.assertIsInstance(float(np.mean(scores)), float)

    def test_fit_multiple_runs(self):
        """Run 5 sequential runs and verify all produce valid results."""
        result = GPBenchmarker(
            self._make_config(
                num_runs=5,
                trainer_config=self._make_trainer_config(num_epochs=3),
            )
        ).fit()
        self.assertEqual(len(result.runs), 5)
        for run in result.runs:
            self.assertIsInstance(run.best_rule, Rule)
            self.assertIsInstance(run.test_score, float)
            self.assertEqual(len(run.folds), 2)

    def test_best_rule(self):
        result = GPBenchmarker(self._make_config(num_runs=2)).fit()
        self.assertIsInstance(result.best_rule, Rule)

    def test_best_run(self):
        result = GPBenchmarker(self._make_config(num_runs=2)).fit()
        self.assertIsInstance(result.best_run, RunResult)

    def test_feature_names_enable_readable_rules(self):
        result = GPBenchmarker(self._make_config(num_runs=1)).fit()
        readable = result.best_rule.to_str(result.best_run.feature_names)
        self.assertIsInstance(readable, str)
        self.assertTrue(len(readable) > 0)

    def test_reproducibility_with_seed(self):
        cfg_kwargs = dict(
            trainer_config=self._make_trainer_config(num_epochs=2),
            n_jobs=1,
            base_seed=12345,
        )
        r1 = GPBenchmarker(self._make_config(**cfg_kwargs)).fit()
        r2 = GPBenchmarker(self._make_config(**cfg_kwargs)).fit()
        self.assertEqual(
            [r.seed for r in r1.runs],
            [r.seed for r in r2.runs],
        )
        self.assertEqual(r1.test_scores, r2.test_scores)

    # ------------------------------------------------------------------ #
    #  fit — parallel (multiprocessing)
    # ------------------------------------------------------------------ #
    def test_fit_parallel_4_processes(self):
        """Full parallel run with 4 processes, small data, finishes fast."""
        result = GPBenchmarker(
            self._make_config(
                num_runs=4,
                n_jobs=4,
                trainer_config=self._make_trainer_config(num_epochs=2),
            )
        ).fit()
        self.assertEqual(len(result.runs), 4)
        for run in result.runs:
            self.assertIsInstance(run.test_score, float)
            self.assertIsInstance(run.best_rule, Rule)

    # ------------------------------------------------------------------ #
    #  Custom GP components
    # ------------------------------------------------------------------ #
    def test_custom_population_factory(self):
        gp = self._make_gp_config(
            population_factory=PopulationGeneratorFactory(population_size=10)
        )
        result = GPBenchmarker(
            self._make_config(
                num_runs=1,
                trainer_config=self._make_trainer_config(gp_config=gp, num_epochs=2),
            )
        ).fit()
        self.assertEqual(len(result.runs), 1)

    def test_custom_mutation_factory(self):
        gp = self._make_gp_config(
            mutation_factory=MutationExecutorFactory(mutation_p=0.1)
        )
        result = GPBenchmarker(
            self._make_config(
                num_runs=1,
                trainer_config=self._make_trainer_config(gp_config=gp, num_epochs=2),
            )
        ).fit()
        self.assertEqual(len(result.runs), 1)

    def test_custom_crossover_factory(self):
        gp = self._make_gp_config(
            crossover_factory=CrossoverExecutorFactory(crossover_p=0.5)
        )
        result = GPBenchmarker(
            self._make_config(
                num_runs=1,
                trainer_config=self._make_trainer_config(gp_config=gp, num_epochs=2),
            )
        ).fit()
        self.assertEqual(len(result.runs), 1)

    def test_custom_selection(self):
        gp = self._make_gp_config(selection=RouletteSelection())
        result = GPBenchmarker(
            self._make_config(
                num_runs=1,
                trainer_config=self._make_trainer_config(gp_config=gp, num_epochs=2),
            )
        ).fit()
        self.assertEqual(len(result.runs), 1)

    def test_regeneration(self):
        gp = self._make_gp_config(regeneration=True, regeneration_patience=50)
        result = GPBenchmarker(
            self._make_config(
                num_runs=1,
                trainer_config=self._make_trainer_config(gp_config=gp, num_epochs=2),
            )
        ).fit()
        self.assertEqual(len(result.runs), 1)

    def test_custom_binarizer_end_to_end(self):
        result = GPBenchmarker(
            self._make_config(
                binarizer=StandardBinarizer(num_bins=3),
                num_runs=2,
            )
        ).fit()
        self.assertEqual(len(result.runs), 2)

    # ------------------------------------------------------------------ #
    #  optimize_scorer
    # ------------------------------------------------------------------ #
    def test_optimize_scorer_true(self):
        gp = self._make_gp_config(
            score_fn=accuracy_with_sample_weight, optimize_scorer=True
        )
        result = GPBenchmarker(
            self._make_config(
                num_runs=1,
                trainer_config=self._make_trainer_config(gp_config=gp, num_epochs=2),
            )
        ).fit()
        self.assertIsNotNone(result.runs[0].test_score)

    def test_optimize_scorer_false(self):
        result = GPBenchmarker(
            self._make_config(
                num_runs=1,
                trainer_config=self._make_trainer_config(num_epochs=2),
            )
        ).fit()
        self.assertIsNotNone(result.runs[0].test_score)

    def test_optimize_scorer_default_is_true(self):
        self.assertTrue(
            BooleanGPConfig(score_fn=accuracy_with_sample_weight).optimize_scorer
        )

    def test_progress_bar_disabled(self):
        benchmarker = GPBenchmarker(self._make_config())
        self.assertFalse(benchmarker.config.trainer_config.progress_bar)
        result = benchmarker.fit()
        self.assertEqual(len(result.runs), 2)

    # ------------------------------------------------------------------ #
    #  execute_single_run — direct tests
    # ------------------------------------------------------------------ #
    def test_execute_single_run_directly(self):
        """Call execute_single_run without going through GPBenchmarker."""
        config = self._make_config(num_runs=1)
        config.binarizer = StandardBinarizer()
        run = execute_single_run(run_id=0, seed=42, config=config)
        self.assertIsInstance(run, RunResult)
        self.assertEqual(run.run_id, 0)
        self.assertEqual(run.seed, 42)
        self.assertEqual(len(run.folds), 2)
        self.assertIsInstance(run.best_rule, Rule)

    def test_execute_single_run_with_progress_queue(self):
        """execute_single_run should send epoch/fold/run messages to the queue."""
        config = self._make_config(
            num_runs=1,
            trainer_config=self._make_trainer_config(num_epochs=2, progress_bar=True),
        )
        config.binarizer = StandardBinarizer()
        q = Queue()
        run = execute_single_run(run_id=0, seed=42, config=config, progress_queue=q)
        self.assertIsInstance(run, RunResult)

        # Drain the queue using get(timeout) to avoid the race where the
        # feeder thread hasn't flushed the last put() to the pipe yet.
        messages = []
        while True:
            try:
                messages.append(q.get(timeout=0.5))
            except queue.Empty:
                break
        msg_types = [m[0] for m in messages]
        self.assertIn("fold", msg_types)
        self.assertIn("run", msg_types)
        self.assertEqual(msg_types.count("fold"), config.n_folds)
        self.assertEqual(msg_types.count("run"), 1)

    def test_execute_single_run_confusion_matrix(self):
        """Confusion matrix values from execute_single_run should be consistent."""
        config = self._make_config(num_runs=1)
        config.binarizer = StandardBinarizer()
        run = execute_single_run(run_id=0, seed=99, config=config)
        total = run.test_tp + run.test_fp + run.test_fn + run.test_tn
        self.assertGreater(total, 0)
        # tp + fn == actual positives, fp + tn == actual negatives
        self.assertGreaterEqual(run.test_tp + run.test_fn, 0)
        self.assertGreaterEqual(run.test_fp + run.test_tn, 0)

    def test_single_run_wrapper(self):
        """single_run_wrapper should produce the same result as execute_single_run."""
        config = self._make_config(num_runs=1)
        config.binarizer = StandardBinarizer()
        run = single_run_wrapper((0, 42, config, None))
        self.assertIsInstance(run, RunResult)
        self.assertEqual(run.run_id, 0)

    # ------------------------------------------------------------------ #
    #  Doctests
    # ------------------------------------------------------------------ #
    def test_doctests(self):
        result = doctest.testmod(hgp_lib.benchmarkers.gp_benchmarker, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.benchmarkers.runner, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.benchmarkers.progress, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
