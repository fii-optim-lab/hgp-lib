"""Tests for all config modules: BooleanGPConfig, TrainerConfig, BenchmarkerConfig."""

import doctest
import unittest

import numpy as np
import pandas as pd

import hgp_lib.configs.boolean_gp_config
import hgp_lib.configs.trainer_config
import hgp_lib.configs.benchmarker_config
from hgp_lib.configs import (
    BenchmarkerConfig,
    BooleanGPConfig,
    TrainerConfig,
    validate_benchmarker_config,
    validate_gp_config,
    validate_trainer_config,
)
from hgp_lib.mutations import MutationExecutorFactory
from hgp_lib.populations import PopulationGeneratorFactory
from hgp_lib.preprocessing import StandardBinarizer


def accuracy(predictions, labels):
    """Module-level score function."""
    return float((predictions == labels).mean())


class TestBooleanGPConfig(unittest.TestCase):
    """Tests for BooleanGPConfig and validate_gp_config."""

    def setUp(self):
        self.data = np.array(
            [[True, False], [False, True], [True, True], [False, False]]
        )
        self.labels = np.array([1, 0, 1, 0])

    def test_valid_config(self):
        config = BooleanGPConfig(
            score_fn=accuracy, train_data=self.data, train_labels=self.labels
        )
        validate_gp_config(config)  # Should not raise

    def test_valid_config_without_data(self):
        """Template config for benchmarker (no data required)."""
        config = BooleanGPConfig(score_fn=accuracy)
        validate_gp_config(config, require_data=False)  # Should not raise

    def test_missing_data_raises(self):
        config = BooleanGPConfig(score_fn=accuracy)
        with self.assertRaises(ValueError):
            validate_gp_config(config, require_data=True)

    def test_score_fn_must_be_callable(self):
        with self.assertRaises(TypeError):
            config = BooleanGPConfig(score_fn="not callable")
            validate_gp_config(config, require_data=False)

    def test_regeneration_patience_must_be_positive(self):
        config = BooleanGPConfig(
            score_fn=accuracy, regeneration=True, regeneration_patience=0
        )
        with self.assertRaises(ValueError):
            validate_gp_config(config, require_data=False)

    def test_max_depth_requires_sampling_strategy(self):
        config = BooleanGPConfig(
            score_fn=accuracy, max_depth=1, num_child_populations=3
        )
        with self.assertRaises(ValueError):
            validate_gp_config(config, require_data=False)

    def test_max_depth_requires_child_populations(self):
        config = BooleanGPConfig(
            score_fn=accuracy, max_depth=1, num_child_populations=0
        )
        with self.assertRaises(ValueError):
            validate_gp_config(config, require_data=False)

    def test_feedback_type_invalid(self):
        config = BooleanGPConfig(score_fn=accuracy, feedback_type="invalid")
        with self.assertRaises(ValueError):
            validate_gp_config(config, require_data=False)

    def test_feedback_strength_must_be_non_negative(self):
        config = BooleanGPConfig(score_fn=accuracy, feedback_strength=-0.1)
        with self.assertRaises(ValueError):
            validate_gp_config(config, require_data=False)

    def test_feedback_strength_zero_is_valid(self):
        config = BooleanGPConfig(score_fn=accuracy, feedback_strength=0)
        validate_gp_config(config, require_data=False)  # Should not raise

    def test_top_k_transfer_must_be_at_least_1(self):
        config = BooleanGPConfig(score_fn=accuracy, top_k_transfer=0)
        with self.assertRaises(ValueError):
            validate_gp_config(config, require_data=False)

    def test_defaults(self):
        config = BooleanGPConfig(score_fn=accuracy)
        self.assertTrue(config.optimize_scorer)
        self.assertFalse(config.regeneration)
        self.assertEqual(config.regeneration_patience, 100)
        self.assertEqual(config.max_depth, 0)
        self.assertEqual(config.num_child_populations, 0)
        self.assertIsInstance(config.population_factory, PopulationGeneratorFactory)
        self.assertIsInstance(config.mutation_factory, MutationExecutorFactory)
        self.assertEqual(config.population_factory.population_size, 100)
        self.assertEqual(config.mutation_factory.mutation_p, 0.1)
        self.assertIsNone(config.crossover_executor)
        self.assertIsNone(config.selection)


class TestTrainerConfig(unittest.TestCase):
    """Tests for TrainerConfig and validate_trainer_config."""

    def setUp(self):
        self.data = np.array(
            [[True, False], [False, True], [True, True], [False, False]]
        )
        self.labels = np.array([1, 0, 1, 0])
        self.gp_config = BooleanGPConfig(
            score_fn=accuracy, train_data=self.data, train_labels=self.labels
        )

    def test_valid_config(self):
        config = TrainerConfig(gp_config=self.gp_config, num_epochs=10)
        validate_trainer_config(config)  # Should not raise

    def test_valid_config_without_data(self):
        gp_config = BooleanGPConfig(score_fn=accuracy)
        config = TrainerConfig(gp_config=gp_config, num_epochs=10)
        validate_trainer_config(config, require_data=False)  # Should not raise

    def test_num_epochs_must_be_int(self):
        config = TrainerConfig(gp_config=self.gp_config, num_epochs=5.0)
        with self.assertRaises(TypeError):
            validate_trainer_config(config)

    def test_num_epochs_must_be_positive(self):
        config = TrainerConfig(gp_config=self.gp_config, num_epochs=0)
        with self.assertRaises(ValueError):
            validate_trainer_config(config)

    def test_val_every_must_be_positive(self):
        config = TrainerConfig(gp_config=self.gp_config, num_epochs=10, val_every=0)
        with self.assertRaises(ValueError):
            validate_trainer_config(config)

    def test_val_data_and_labels_must_both_be_provided(self):
        config = TrainerConfig(
            gp_config=self.gp_config,
            num_epochs=10,
            val_data=self.data,
            val_labels=None,
        )
        with self.assertRaises(ValueError):
            validate_trainer_config(config)

    def test_valid_with_validation_data(self):
        config = TrainerConfig(
            gp_config=self.gp_config,
            num_epochs=10,
            val_data=self.data,
            val_labels=self.labels,
        )
        validate_trainer_config(config)  # Should not raise

    def test_defaults(self):
        config = TrainerConfig(gp_config=self.gp_config, num_epochs=10)
        self.assertEqual(config.val_every, 100)
        self.assertTrue(config.progress_bar)
        self.assertTrue(config.leave_progress_bar)
        self.assertIsNone(config.progress_callback)
        self.assertEqual(config.progress_update_interval, 100)


class TestBenchmarkerConfig(unittest.TestCase):
    """Tests for BenchmarkerConfig and validate_benchmarker_config."""

    def setUp(self):
        self.data = pd.DataFrame(
            {
                "feature1": [True, False, True, False, True, False, True, False],
                "feature2": [False, True, True, False, False, True, True, False],
            }
        )
        self.labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        self.gp_config = BooleanGPConfig(score_fn=accuracy)
        self.trainer_config = TrainerConfig(gp_config=self.gp_config, num_epochs=5)

    def _make_config(self, **kwargs):
        defaults = dict(
            data=self.data,
            labels=self.labels,
            trainer_config=self.trainer_config,
            n_folds=2,
        )
        defaults.update(kwargs)
        return BenchmarkerConfig(**defaults)

    def test_valid_config(self):
        config = self._make_config()
        validate_benchmarker_config(config)  # Should not raise

    def test_data_must_be_dataframe(self):
        with self.assertRaises(TypeError):
            config = self._make_config(data=self.data.to_numpy())
            validate_benchmarker_config(config)

    def test_data_string_rejected(self):
        with self.assertRaises(TypeError):
            config = self._make_config(data="not a dataframe")
            validate_benchmarker_config(config)

    def test_labels_must_be_ndarray(self):
        with self.assertRaises(TypeError):
            config = self._make_config(labels="not array")
            validate_benchmarker_config(config)

    def test_labels_length_must_match(self):
        with self.assertRaises(ValueError):
            config = self._make_config(labels=np.array([1, 0]))
            validate_benchmarker_config(config)

    def test_num_runs_must_be_positive(self):
        with self.assertRaises(ValueError):
            config = self._make_config(num_runs=0)
            validate_benchmarker_config(config)

    def test_test_size_must_be_in_range(self):
        with self.assertRaises(ValueError):
            config = self._make_config(test_size=0.0)
            validate_benchmarker_config(config)
        with self.assertRaises(ValueError):
            config = self._make_config(test_size=1.0)
            validate_benchmarker_config(config)

    def test_n_folds_must_be_at_least_2(self):
        with self.assertRaises(ValueError):
            config = self._make_config(n_folds=1)
            validate_benchmarker_config(config)

    def test_fitted_binarizer_rejected(self):
        binarizer = StandardBinarizer(num_bins=2)
        binarizer.fit_transform(self.data)
        with self.assertRaises(ValueError):
            config = self._make_config(binarizer=binarizer)
            validate_benchmarker_config(config)

    def test_custom_binarizer_accepted(self):
        binarizer = StandardBinarizer(num_bins=3)
        config = self._make_config(binarizer=binarizer)
        validate_benchmarker_config(config)  # Should not raise

    def test_invalid_binarizer_type(self):
        with self.assertRaises(TypeError):
            config = self._make_config(binarizer="not a binarizer")
            validate_benchmarker_config(config)

    def test_defaults(self):
        config = self._make_config()
        self.assertEqual(config.num_runs, 30)
        self.assertEqual(config.test_size, 0.2)
        self.assertEqual(config.n_jobs, -1)
        self.assertEqual(config.base_seed, 0)
        self.assertTrue(config.show_run_progress)
        self.assertTrue(config.show_fold_progress)
        self.assertTrue(config.show_epoch_progress)
        self.assertIsNone(config.binarizer)


class TestConfigDoctests(unittest.TestCase):
    """Run doctests for all config modules."""

    def test_boolean_gp_config_doctests(self):
        result = doctest.testmod(hgp_lib.configs.boolean_gp_config, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")

    def test_trainer_config_doctests(self):
        result = doctest.testmod(hgp_lib.configs.trainer_config, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")

    def test_benchmarker_config_doctests(self):
        result = doctest.testmod(hgp_lib.configs.benchmarker_config, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


class TestComplexityCheck(unittest.TestCase):
    """Tests for complexity_check helper function."""

    def test_complexity_check_accepts_small_rules(self):
        from hgp_lib.utils import complexity_check
        from hgp_lib.rules import Literal, And, Or

        check = complexity_check(5)
        self.assertTrue(check(Literal(value=0)))  # len=1
        self.assertTrue(check(And([Literal(value=0), Literal(value=1)])))  # len=3
        self.assertTrue(
            check(And([Literal(value=0), Or([Literal(value=1), Literal(value=2)])]))
        )  # len=5

    def test_complexity_check_rejects_large_rules(self):
        from hgp_lib.utils import complexity_check
        from hgp_lib.rules import Literal, And, Or

        check = complexity_check(3)
        self.assertFalse(
            check(And([Literal(value=0), Or([Literal(value=1), Literal(value=2)])]))
        )  # len=5


if __name__ == "__main__":
    unittest.main()
