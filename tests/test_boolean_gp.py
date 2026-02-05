import doctest
import unittest
import random

import numpy as np

import hgp_lib.algorithms.boolean_gp
from hgp_lib.algorithms import BooleanGP
from hgp_lib.configs import BooleanGPConfig
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.mutations import (
    MutationExecutor,
    create_standard_literal_mutations,
    create_standard_operator_mutations,
)
from hgp_lib.populations import PopulationGenerator, RandomStrategy
from hgp_lib.rules import Rule
from hgp_lib.selections import TournamentSelection


class TestBooleanGP(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        np.random.seed(42)

        self.train_data = np.array(
            [
                [True, False, True, False],
                [False, True, False, True],
                [True, True, False, False],
                [False, False, True, True],
            ]
        )
        self.train_labels = np.array([1, 0, 1, 0])
        self.val_data = np.array(
            [[True, False, False, True], [False, True, True, False]]
        )
        self.val_labels = np.array([1, 0])
        self.num_features = 4

        def accuracy(predictions, labels):
            return np.mean(predictions == labels)

        self.score_fn = accuracy

        self.generator = PopulationGenerator(
            strategies=[RandomStrategy(num_literals=self.num_features)],
            population_size=10,
        )

        self.mutation_executor = MutationExecutor(
            literal_mutations=create_standard_literal_mutations(self.num_features),
            operator_mutations=create_standard_operator_mutations(self.num_features),
        )

    def _make_config(self, **kwargs):
        """Helper to create BooleanGPConfig with test defaults.

        Note: optimize_scorer=False by default since test scorer doesn't support sample_weight.
        """
        defaults = dict(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
            optimize_scorer=False,
        )
        defaults.update(kwargs)
        return BooleanGPConfig(**defaults)

    def test_boolean_gp_validation(self):
        with self.subTest("score_fn must be callable"):
            with self.assertRaises(TypeError):
                config = self._make_config(score_fn="not callable")
                BooleanGP(config)

        with self.subTest("population_generator must be PopulationGenerator"):
            with self.assertRaises(TypeError):
                config = self._make_config(population_generator="not generator")
                BooleanGP(config)

        with self.subTest("mutation_executor must be MutationExecutor"):
            with self.assertRaises(TypeError):
                config = self._make_config(mutation_executor="not executor")
                BooleanGP(config)

        with self.subTest("crossover_executor must be CrossoverExecutor"):
            with self.assertRaises(TypeError):
                config = self._make_config(crossover_executor="not executor")
                BooleanGP(config)

        with self.subTest("selection must be BaseSelection"):
            with self.assertRaises(TypeError):
                config = self._make_config(selection="not selection")
                BooleanGP(config)

        with self.subTest("regeneration must be bool"):
            with self.assertRaises(TypeError):
                config = self._make_config(regeneration=1)
                BooleanGP(config)

        with self.subTest("regeneration_patience must be int"):
            with self.assertRaises(TypeError):
                config = self._make_config(regeneration_patience=1.5)
                BooleanGP(config)

        with self.subTest(
            "regeneration_patience must be positive when regeneration=True"
        ):
            with self.assertRaises(ValueError):
                config = self._make_config(regeneration=True, regeneration_patience=0)
                BooleanGP(config)

    def test_boolean_gp_init(self):
        config = self._make_config()
        gp = BooleanGP(config)

        self.assertIsNotNone(gp.population)
        self.assertEqual(len(gp.population), 10)
        self.assertEqual(gp.best_score, -float("inf"))
        self.assertIsNone(gp.best_rule)
        self.assertEqual(gp.real_best_score, -float("inf"))
        self.assertIsNone(gp.real_best_rule)
        self.assertEqual(gp.best_not_improved_epochs, 0)

    def test_boolean_gp_defaults(self):
        config = self._make_config()
        gp = BooleanGP(config)

        self.assertIsInstance(gp.crossover_executor, CrossoverExecutor)

        self.assertIsInstance(gp.selection, TournamentSelection)

    def test_step_returns_metrics(self):
        config = self._make_config()
        gp = BooleanGP(config)

        metrics = gp.step()

        self.assertIsNotNone(metrics.best)
        self.assertIsNotNone(metrics.best_rule)
        self.assertIsNotNone(metrics.real_best)
        self.assertIsNotNone(metrics.real_best_rule)
        self.assertIsNotNone(metrics.current_best)
        self.assertIsNotNone(metrics.population_scores)
        self.assertIsNotNone(metrics.epoch)
        self.assertIsNotNone(metrics.best_not_improved_epochs)
        self.assertIsNotNone(metrics.regenerated)

        self.assertEqual(metrics.epoch, 0)
        self.assertIsInstance(metrics.best_rule, Rule)
        self.assertIsInstance(metrics.real_best_rule, Rule)
        self.assertIsInstance(metrics.population_scores, np.ndarray)
        self.assertGreater(len(metrics.population_scores), 0)

    def test_step_updates_best_rule(self):
        config = self._make_config()
        gp = BooleanGP(config)

        initial_best = gp.best_score
        metrics = gp.step()

        self.assertGreaterEqual(metrics.best, initial_best)
        self.assertIsNotNone(gp.best_rule)
        self.assertIsNotNone(gp.real_best_rule)

    def test_step_increments_epoch(self):
        config = self._make_config()
        gp = BooleanGP(config)

        metrics1 = gp.step()
        self.assertEqual(metrics1.epoch, 0)

        metrics2 = gp.step()
        self.assertEqual(metrics2.epoch, 1)

    def test_step_updates_population_size(self):
        config = self._make_config()
        gp = BooleanGP(config)

        initial_size = len(gp.population)
        gp.step()

        self.assertEqual(len(gp.population), initial_size)

    def test_validate_best_raises_without_steps(self):
        config = self._make_config()
        gp = BooleanGP(config)

        with self.assertRaises(RuntimeError) as context:
            gp.validate_best(self.val_data, self.val_labels)

        self.assertIn("No best rule available", str(context.exception))

    def test_validate_best_returns_metrics(self):
        config = self._make_config()
        gp = BooleanGP(config)

        gp.step()
        metrics = gp.validate_best(self.val_data, self.val_labels)

        self.assertIsNotNone(metrics.best)
        self.assertIsNotNone(metrics.best_rule)
        self.assertIsInstance(metrics.best, float)
        self.assertIsInstance(metrics.best_rule, Rule)

    def test_validate_best_all_time_best(self):
        config = self._make_config()
        gp = BooleanGP(config)

        gp.step()
        current_metrics = gp.validate_best(
            self.val_data, self.val_labels, all_time_best=False
        )
        all_time_metrics = gp.validate_best(
            self.val_data, self.val_labels, all_time_best=True
        )

        self.assertIsInstance(current_metrics.best_rule, Rule)
        self.assertIsInstance(all_time_metrics.best_rule, Rule)

    def test_validate_best_custom_score_fn(self):
        config = self._make_config()
        gp = BooleanGP(config)

        gp.step()

        def custom_score(predictions, labels):
            return np.sum(predictions & labels)

        metrics = gp.validate_best(
            self.val_data, self.val_labels, score_fn=custom_score
        )
        self.assertIsNotNone(metrics.best)

    def test_validate_population_raises_without_steps(self):
        config = self._make_config()
        gp = BooleanGP(config)

        with self.assertRaises(RuntimeError) as context:
            gp.validate_population(self.val_data, self.val_labels)

        self.assertIn("No best rule available", str(context.exception))

    def test_validate_population_returns_metrics(self):
        config = self._make_config()
        gp = BooleanGP(config)

        gp.step()
        metrics = gp.validate_population(self.val_data, self.val_labels)

        self.assertIsNotNone(metrics.best)
        self.assertIsNotNone(metrics.best_rule)
        self.assertIsNotNone(metrics.population_scores)
        self.assertIsInstance(metrics.best, float)
        self.assertIsInstance(metrics.best_rule, Rule)
        self.assertEqual(len(metrics.population_scores), len(gp.population))

    def test_validate_population_all_time_best(self):
        config = self._make_config()
        gp = BooleanGP(config)

        gp.step()
        current_metrics = gp.validate_population(
            self.val_data, self.val_labels, all_time_best=False
        )
        all_time_metrics = gp.validate_population(
            self.val_data, self.val_labels, all_time_best=True
        )

        self.assertIsInstance(current_metrics.best_rule, Rule)
        self.assertIsInstance(all_time_metrics.best_rule, Rule)

    def test_validate_population_custom_score_fn(self):
        config = self._make_config()
        gp = BooleanGP(config)

        gp.step()

        def custom_score(predictions, labels):
            return np.sum(predictions & labels)

        metrics = gp.validate_population(
            self.val_data, self.val_labels, score_fn=custom_score
        )
        self.assertIsNotNone(metrics.best)
        self.assertIsNotNone(metrics.population_scores)

    def test_regeneration_disabled(self):
        config = self._make_config(regeneration=False, regeneration_patience=1)
        gp = BooleanGP(config)

        gp.step()

        self.assertFalse(gp.regeneration)

    def test_regeneration_enabled(self):
        config = self._make_config(regeneration=True, regeneration_patience=1)
        gp = BooleanGP(config)

        gp.step()

        metrics = gp.step()

        if metrics.regenerated:
            self.assertEqual(gp.best_score, -float("inf"))
            self.assertEqual(gp.best_not_improved_epochs, 0)

    def test_best_not_improved_epochs_tracking(self):
        config = self._make_config()
        gp = BooleanGP(config)

        metrics1 = gp.step()
        initial_epochs = metrics1.best_not_improved_epochs

        metrics2 = gp.step()

        if metrics2.current_best < metrics1.best:
            self.assertGreater(metrics2.best_not_improved_epochs, initial_epochs)
        else:
            self.assertEqual(metrics2.best_not_improved_epochs, 0)

    def test_real_best_tracking(self):
        config = self._make_config()
        gp = BooleanGP(config)

        gp.step()
        initial_real_best = gp.real_best_score

        gp.step()

        self.assertGreaterEqual(gp.real_best_score, initial_real_best)
        if gp.best_score > initial_real_best:
            self.assertEqual(gp.real_best_score, gp.best_score)

    def test_multiple_steps(self):
        config = self._make_config()
        gp = BooleanGP(config)

        for i in range(5):
            metrics = gp.step()
            self.assertEqual(metrics.epoch, i)
            self.assertIsNotNone(metrics.best_rule)

    def test_step_with_custom_crossover(self):
        crossover_executor = CrossoverExecutor(crossover_p=0.5)
        config = self._make_config(crossover_executor=crossover_executor)
        gp = BooleanGP(config)

        metrics = gp.step()
        self.assertIsNotNone(metrics.best)

    def test_step_with_custom_selection(self):
        from hgp_lib.selections import RouletteSelection

        selection = RouletteSelection()
        config = self._make_config(selection=selection)
        gp = BooleanGP(config)

        metrics = gp.step()
        self.assertIsNotNone(metrics.best)

    def test_optimize_scorer_true(self):
        """Test that optimize_scorer=True works with a scorer supporting sample_weight."""

        def accuracy_with_weight(predictions, labels, sample_weight=None):
            if sample_weight is None:
                return np.mean(predictions == labels)
            correct = predictions == labels
            return np.dot(correct, sample_weight) / sample_weight.sum()

        config = self._make_config(score_fn=accuracy_with_weight, optimize_scorer=True)
        gp = BooleanGP(config)

        metrics = gp.step()
        self.assertIsNotNone(metrics.best)
        self.assertIsInstance(metrics.best, float)

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.algorithms.boolean_gp, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
