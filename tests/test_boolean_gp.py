import doctest
import unittest
import random
from typing import Sequence

import numpy as np

import hgp_lib.algorithms.boolean_gp
from hgp_lib.algorithms import BooleanGP
from hgp_lib.configs import BooleanGPConfig
from hgp_lib.crossover import CrossoverExecutor, CrossoverExecutorFactory
from hgp_lib.populations import PopulationGeneratorFactory, FeatureSamplingStrategy
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

    def _make_config(self, **kwargs):
        defaults = dict(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_factory=PopulationGeneratorFactory(population_size=10),
            optimize_scorer=False,
        )
        defaults.update(kwargs)
        return BooleanGPConfig(**defaults)

    def test_boolean_gp_validation(self):
        with self.subTest("score_fn must be callable"):
            with self.assertRaises(TypeError):
                config = self._make_config(score_fn="not callable")
                BooleanGP(config)

        with self.subTest("population_factory must be PopulationGeneratorFactory"):
            with self.assertRaises(TypeError):
                config = self._make_config(population_factory="not factory")
                BooleanGP(config)

        with self.subTest("mutation_factory must be MutationExecutorFactory"):
            with self.assertRaises(TypeError):
                config = self._make_config(mutation_factory="not factory")
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

        self.assertIsNotNone(metrics.best_train_score)
        self.assertIsNotNone(metrics.best_rule)
        self.assertIsNotNone(metrics.train_scores)
        self.assertIsNotNone(metrics.complexities)

        self.assertIsInstance(metrics.best_rule, Rule)
        self.assertIsInstance(metrics.train_scores, Sequence)
        self.assertGreater(len(metrics.train_scores), 0)

    def test_step_updates_best_rule(self):
        config = self._make_config()
        gp = BooleanGP(config)

        initial_best = gp.best_score
        metrics = gp.step()

        self.assertGreaterEqual(metrics.best_train_score, initial_best)
        self.assertIsNotNone(gp.best_rule)

    def test_step_updates_population_size(self):
        config = self._make_config()
        gp = BooleanGP(config)

        initial_size = len(gp.population)
        gp.step()

        self.assertEqual(len(gp.population), initial_size)

    def test_evaluate_best(self):
        config = self._make_config()
        gp = BooleanGP(config)

        gp.step()
        score = gp.evaluate_best(self.val_data, self.val_labels)

        self.assertIsInstance(score, float)

    def test_evaluate_best_custom_score_fn(self):
        config = self._make_config()
        gp = BooleanGP(config)

        gp.step()

        def custom_score(predictions, labels):
            return float(np.sum(predictions & labels))

        score = gp.evaluate_best(self.val_data, self.val_labels, score_fn=custom_score)
        self.assertIsInstance(score, float)

    def test_regeneration_disabled(self):
        config = self._make_config(regeneration=False, regeneration_patience=1)
        gp = BooleanGP(config)

        gp.step()

        self.assertFalse(gp.regeneration)

    def test_regeneration_enabled(self):
        config = self._make_config(regeneration=True, regeneration_patience=1)
        gp = BooleanGP(config)

        gp.step()
        gp.step()

    def test_best_not_improved_epochs_tracking(self):
        config = self._make_config()
        gp = BooleanGP(config)

        gp.step()
        gp.step()

        self.assertIsInstance(gp.best_not_improved_epochs, int)

    def test_multiple_steps(self):
        config = self._make_config()
        gp = BooleanGP(config)

        for i in range(5):
            metrics = gp.step()
            self.assertIsNotNone(metrics.best_rule)

    def test_step_with_custom_crossover(self):
        crossover_factory = CrossoverExecutorFactory(crossover_p=0.5)
        config = self._make_config(crossover_factory=crossover_factory)
        gp = BooleanGP(config)

        metrics = gp.step()
        self.assertIsNotNone(metrics.best_train_score)

    def test_step_with_custom_selection(self):
        from hgp_lib.selections import RouletteSelection

        selection = RouletteSelection()
        config = self._make_config(selection=selection)
        gp = BooleanGP(config)

        metrics = gp.step()
        self.assertIsNotNone(metrics.best_train_score)

    def test_optimize_scorer_true(self):
        def accuracy_with_weight(predictions, labels, sample_weight=None):
            if sample_weight is None:
                return np.mean(predictions == labels)
            correct = predictions == labels
            return np.dot(correct, sample_weight) / sample_weight.sum()

        config = self._make_config(score_fn=accuracy_with_weight, optimize_scorer=True)
        gp = BooleanGP(config)

        metrics = gp.step()
        self.assertIsNotNone(metrics.best_train_score)
        self.assertIsInstance(metrics.best_train_score, float)

    def test_doctests(self):
        # TODO: Check if we can't make a single test file that tests the doctests for all files automatically
        # This means that it will automatically test it, without needing to specify the module
        result = doctest.testmod(hgp_lib.algorithms.boolean_gp, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")

    def test_compute_regularized_scores_formula(self):
        """Test that regularized scores follow the formula: score - penalty * ln(complexity)."""
        config = self._make_config(complexity_penalty=0.0)
        gp = BooleanGP(config)

        for seed in range(10):
            np.random.seed(seed)
            random.seed(seed)

            scores = np.random.uniform(-1.0, 1.0, len(gp.population))
            complexities = [len(rule) for rule in gp.population]

            gp.complexity_penalty = 0.0
            regularized = gp._compute_regularized_scores(scores.copy(), complexities)
            self.assertTrue(
                np.allclose(regularized, scores),
                "When complexity_penalty=0, regularized scores should equal original scores",
            )

            for penalty in [0.001, 0.01, 0.05, 0.1]:
                gp.complexity_penalty = penalty
                regularized = gp._compute_regularized_scores(
                    scores.copy(), complexities
                )
                expected = scores - penalty * np.log(np.array(complexities))

                self.assertTrue(
                    np.allclose(regularized, expected),
                    f"Regularized score formula failed for penalty={penalty}",
                )

    def test_top_k_equals_population_size(self):
        config = self._make_config(
            population_factory=PopulationGeneratorFactory(population_size=100),
            top_k_transfer=100,
            num_child_populations=2,
            max_depth=2,
            sampling_strategy=FeatureSamplingStrategy(),
        )
        gp = BooleanGP(config)

        gp.step()

    def test_step_with_complexity_penalty(self):
        """Test that step() works with complexity_penalty > 0."""
        config = self._make_config(complexity_penalty=0.01)
        gp = BooleanGP(config)

        metrics = gp.step()
        self.assertIsNotNone(metrics.train_scores)
        self.assertIsNotNone(metrics.complexities)

    def test_step_without_complexity_penalty(self):
        """Test that step() works with complexity_penalty = 0."""
        config = self._make_config(complexity_penalty=0.0)
        gp = BooleanGP(config)

        metrics = gp.step()
        self.assertIsNotNone(metrics.train_scores)


if __name__ == "__main__":
    unittest.main()
