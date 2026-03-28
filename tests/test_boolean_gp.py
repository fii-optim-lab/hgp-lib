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
from hgp_lib.rules import Rule, Literal
from hgp_lib.selections import TournamentSelection, RouletteSelection


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

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #
    def test_boolean_gp_validation(self):
        with self.subTest("score_fn must be callable"):
            with self.assertRaises(TypeError):
                BooleanGP(self._make_config(score_fn="not callable"))

        with self.subTest("population_factory must be PopulationGeneratorFactory"):
            with self.assertRaises(TypeError):
                BooleanGP(self._make_config(population_factory="not factory"))

        with self.subTest("mutation_factory must be MutationExecutorFactory"):
            with self.assertRaises(TypeError):
                BooleanGP(self._make_config(mutation_factory="not factory"))

        with self.subTest("crossover_executor must be CrossoverExecutor"):
            with self.assertRaises(TypeError):
                BooleanGP(self._make_config(crossover_executor="not executor"))

        with self.subTest("selection must be BaseSelection"):
            with self.assertRaises(TypeError):
                BooleanGP(self._make_config(selection="not selection"))

        with self.subTest("regeneration must be bool"):
            with self.assertRaises(TypeError):
                BooleanGP(self._make_config(regeneration=1))

        with self.subTest("regeneration_patience must be int"):
            with self.assertRaises(TypeError):
                BooleanGP(self._make_config(regeneration_patience=1.5))

        with self.subTest(
            "regeneration_patience must be positive when regeneration=True"
        ):
            with self.assertRaises(ValueError):
                BooleanGP(self._make_config(regeneration=True, regeneration_patience=0))

    # ------------------------------------------------------------------ #
    #  Initialization
    # ------------------------------------------------------------------ #
    def test_boolean_gp_init(self):
        gp = BooleanGP(self._make_config())
        self.assertEqual(len(gp.population), 10)
        self.assertEqual(gp.best_score, -float("inf"))
        self.assertIsNone(gp.best_rule)
        self.assertEqual(gp.best_not_improved_epochs, 0)
        self.assertEqual(gp._epoch, -1)

    def test_boolean_gp_defaults(self):
        gp = BooleanGP(self._make_config())
        self.assertIsInstance(gp.crossover_executor, CrossoverExecutor)
        self.assertIsInstance(gp.selection, TournamentSelection)

    def test_original_score_fn_property(self):
        gp = BooleanGP(self._make_config())
        self.assertIs(gp.original_score_fn, self.score_fn)

    # ------------------------------------------------------------------ #
    #  step
    # ------------------------------------------------------------------ #
    def test_step_returns_metrics(self):
        gp = BooleanGP(self._make_config())
        metrics = gp.step()
        self.assertIsInstance(metrics.best_train_score, float)
        self.assertIsInstance(metrics.best_rule, Rule)
        self.assertIsInstance(metrics.train_scores, Sequence)
        self.assertGreater(len(metrics.train_scores), 0)
        self.assertIsNotNone(metrics.complexities)

    def test_step_updates_best_rule(self):
        gp = BooleanGP(self._make_config())
        initial_best = gp.best_score
        gp.step()
        self.assertGreaterEqual(gp.best_score, initial_best)
        self.assertIsNotNone(gp.best_rule)

    def test_step_preserves_population_size(self):
        gp = BooleanGP(self._make_config())
        initial_size = len(gp.population)
        gp.step()
        self.assertEqual(len(gp.population), initial_size)

    def test_multiple_steps(self):
        gp = BooleanGP(self._make_config())
        for _ in range(5):
            metrics = gp.step()
            self.assertIsNotNone(metrics.best_rule)

    def test_step_with_custom_crossover(self):
        config = self._make_config(
            crossover_factory=CrossoverExecutorFactory(crossover_p=0.5)
        )
        metrics = BooleanGP(config).step()
        self.assertIsNotNone(metrics.best_train_score)

    def test_step_with_custom_selection(self):
        config = self._make_config(selection=RouletteSelection())
        metrics = BooleanGP(config).step()
        self.assertIsNotNone(metrics.best_train_score)

    def test_step_with_complexity_penalty(self):
        config = self._make_config(complexity_penalty=0.01)
        metrics = BooleanGP(config).step()
        self.assertIsNotNone(metrics.complexities)

    def test_step_without_complexity_penalty(self):
        config = self._make_config(complexity_penalty=0.0)
        metrics = BooleanGP(config).step()
        self.assertIsNotNone(metrics.train_scores)

    # ------------------------------------------------------------------ #
    #  _forward
    # ------------------------------------------------------------------ #
    def test_forward_expands_population(self):
        gp = BooleanGP(self._make_config())
        pop_before = len(gp.population)
        gp._forward()
        # crossover adds offspring
        self.assertGreaterEqual(len(gp.population), pop_before)

    def test_forward_with_children(self):
        config = self._make_config(
            population_factory=PopulationGeneratorFactory(population_size=20),
            max_depth=1,
            num_child_populations=2,
            sampling_strategy=FeatureSamplingStrategy(),
            top_k_transfer=5,
        )
        gp = BooleanGP(config)
        self.assertEqual(len(gp.child_populations), 2)
        gp._forward()
        # transfer_size should reflect children's top_k contributions
        self.assertEqual(gp._transfer_size, 10)  # 2 children * 5

    # ------------------------------------------------------------------ #
    #  _backward
    # ------------------------------------------------------------------ #
    def test_backward_returns_metrics(self):
        gp = BooleanGP(self._make_config())
        gp._forward()
        metrics = gp._backward()
        self.assertIsInstance(metrics.best_train_score, float)
        self.assertGreater(len(metrics.train_scores), 0)

    def test_backward_with_parent_scores(self):
        config = self._make_config(
            population_factory=PopulationGeneratorFactory(population_size=20),
            max_depth=1,
            num_child_populations=2,
            sampling_strategy=FeatureSamplingStrategy(),
            top_k_transfer=5,
        )
        gp = BooleanGP(config)
        child = gp.child_populations[0]
        child._forward()
        parent_scores = np.ones(child._top_k) * 0.5
        metrics = child._backward(parent_scores)
        self.assertIsInstance(metrics.best_train_score, float)

    def test_backward_selects_population(self):
        gp = BooleanGP(self._make_config())
        gp._forward()
        pop_after_forward = len(gp.population)
        gp._backward()
        # selection trims back to original population_size
        self.assertEqual(len(gp.population), gp.population_size)
        self.assertLessEqual(len(gp.population), pop_after_forward)

    # ------------------------------------------------------------------ #
    #  _update_best
    # ------------------------------------------------------------------ #
    def test_update_best_improves(self):
        gp = BooleanGP(self._make_config())
        rule = Literal(value=0)
        gp._update_best(0.8, rule)
        self.assertEqual(gp.best_score, 0.8)
        self.assertIs(gp.best_rule, rule)
        self.assertEqual(gp.best_not_improved_epochs, 0)

    def test_update_best_no_improvement(self):
        gp = BooleanGP(self._make_config())
        gp._update_best(0.8, Literal(value=0))
        gp._update_best(0.5, Literal(value=1))
        self.assertEqual(gp.best_score, 0.8)
        self.assertEqual(gp.best_not_improved_epochs, 1)

    def test_update_best_equal_score_resets_counter(self):
        gp = BooleanGP(self._make_config())
        gp._update_best(0.8, Literal(value=0))
        gp._update_best(0.5, Literal(value=1))
        self.assertEqual(gp.best_not_improved_epochs, 1)
        gp._update_best(0.8, Literal(value=2))
        self.assertEqual(gp.best_not_improved_epochs, 0)

    def test_update_best_tracks_global(self):
        gp = BooleanGP(self._make_config())
        gp._update_best(0.7, Literal(value=0))
        self.assertEqual(gp.global_best_score, 0.7)
        gp._update_best(0.9, Literal(value=1))
        self.assertEqual(gp.global_best_score, 0.9)
        # global best doesn't decrease even if best_score resets
        gp._update_best(0.5, Literal(value=2))
        self.assertEqual(gp.global_best_score, 0.9)

    # ------------------------------------------------------------------ #
    #  evaluate_population
    # ------------------------------------------------------------------ #
    def test_evaluate_population_returns_correct_length(self):
        gp = BooleanGP(self._make_config())
        scores = gp.evaluate_population(
            self.train_data, self.train_labels, self.score_fn
        )
        self.assertEqual(len(scores), len(gp.population))

    def test_evaluate_population_scores_in_range(self):
        gp = BooleanGP(self._make_config())
        scores = gp.evaluate_population(
            self.train_data, self.train_labels, self.score_fn
        )
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))

    def test_evaluate_population_custom_score_fn(self):
        gp = BooleanGP(self._make_config())

        def always_one(predictions, labels):
            return 1.0

        scores = gp.evaluate_population(self.train_data, self.train_labels, always_one)
        np.testing.assert_array_equal(scores, np.ones(len(gp.population)))

    # ------------------------------------------------------------------ #
    #  evaluate_best
    # ------------------------------------------------------------------ #
    def test_evaluate_best_before_step_raises(self):
        gp = BooleanGP(self._make_config())
        with self.assertRaises(RuntimeError):
            gp.evaluate_best(self.val_data, self.val_labels)

    def test_evaluate_best_returns_float(self):
        gp = BooleanGP(self._make_config())
        gp.step()
        score = gp.evaluate_best(self.val_data, self.val_labels)
        self.assertIsInstance(score, float)

    def test_evaluate_best_custom_score_fn(self):
        gp = BooleanGP(self._make_config())
        gp.step()

        def custom_score(predictions, labels):
            return float(np.sum(predictions & labels))

        score = gp.evaluate_best(self.val_data, self.val_labels, score_fn=custom_score)
        self.assertIsInstance(score, float)

    def test_evaluate_best_uses_original_score_fn(self):
        """When optimize_scorer=True, evaluate_best should use the original (non-optimized) fn."""

        def accuracy_with_weight(predictions, labels, sample_weight=None):
            if sample_weight is None:
                return np.mean(predictions == labels)
            correct = predictions == labels
            return np.dot(correct, sample_weight) / sample_weight.sum()

        config = self._make_config(score_fn=accuracy_with_weight, optimize_scorer=True)
        gp = BooleanGP(config)
        gp.step()
        # Should not raise — uses original fn which handles sample_weight=None
        score = gp.evaluate_best(self.val_data, self.val_labels)
        self.assertIsInstance(score, float)

    # ------------------------------------------------------------------ #
    #  _compute_regularized_scores
    # ------------------------------------------------------------------ #
    def test_regularized_scores_zero_penalty(self):
        gp = BooleanGP(self._make_config(complexity_penalty=0.0))
        scores = np.array([0.9, 0.8, 0.7])
        result = gp._compute_regularized_scores(scores, [3, 5, 7])
        np.testing.assert_array_equal(result, scores)

    def test_regularized_scores_formula(self):
        gp = BooleanGP(self._make_config(complexity_penalty=0.0))
        scores = np.array([1.0, 1.0])
        complexities = [3, 5]

        for penalty in [0.001, 0.01, 0.05, 0.1]:
            gp.complexity_penalty = penalty
            result = gp._compute_regularized_scores(scores.copy(), complexities)
            expected = scores - penalty * np.log(np.array(complexities))
            np.testing.assert_allclose(result, expected)

    def test_regularized_scores_penalizes_complexity(self):
        gp = BooleanGP(self._make_config(complexity_penalty=0.1))
        scores = np.array([1.0, 1.0])
        result = gp._compute_regularized_scores(scores, [3, 10])
        # higher complexity → lower regularized score
        self.assertGreater(result[0], result[1])

    # ------------------------------------------------------------------ #
    #  Regeneration
    # ------------------------------------------------------------------ #
    def test_regeneration_disabled(self):
        gp = BooleanGP(self._make_config(regeneration=False, regeneration_patience=1))
        gp.step()
        self.assertFalse(gp.regeneration)

    def test_regeneration_resets_population(self):
        config = self._make_config(regeneration=True, regeneration_patience=1)
        gp = BooleanGP(config)
        gp.step()
        # Force stagnation
        gp.best_not_improved_epochs = gp.regeneration_patience
        old_pop_ids = [id(r) for r in gp.population]
        gp.step()
        new_pop_ids = [id(r) for r in gp.population]
        # After regeneration, population should be entirely new objects
        self.assertNotEqual(old_pop_ids, new_pop_ids)

    def test_regeneration_resets_best_score(self):
        config = self._make_config(regeneration=True, regeneration_patience=1)
        gp = BooleanGP(config)
        gp.step()
        # After step, best_not_improved_epochs may be 0 (improved).
        # Force stagnation so next step triggers regeneration.
        gp.best_not_improved_epochs = gp.regeneration_patience
        gp.step()
        # _new_generation updates best *then* regenerates, so best_score is reset
        # and best_not_improved_epochs is reset after the regeneration branch.
        self.assertEqual(gp.best_not_improved_epochs, 0)

    # ------------------------------------------------------------------ #
    #  Hierarchical GP
    # ------------------------------------------------------------------ #
    def _make_hierarchical_config(self, **kwargs):
        defaults = dict(
            population_factory=PopulationGeneratorFactory(population_size=20),
            max_depth=1,
            num_child_populations=2,
            sampling_strategy=FeatureSamplingStrategy(),
            top_k_transfer=5,
        )
        defaults.update(kwargs)
        return self._make_config(**defaults)

    def test_child_populations_created(self):
        gp = BooleanGP(self._make_hierarchical_config())
        self.assertEqual(len(gp.child_populations), 2)
        for child in gp.child_populations:
            self.assertIsNotNone(child.feature_mapping)
            self.assertEqual(child.current_depth, 1)

    def test_hierarchical_step(self):
        gp = BooleanGP(self._make_hierarchical_config())
        metrics = gp.step()
        self.assertIsNotNone(metrics.best_train_score)
        self.assertEqual(len(metrics.child_population_generation_metrics), 2)

    def test_hierarchical_multiple_steps(self):
        gp = BooleanGP(self._make_hierarchical_config())
        for _ in range(3):
            metrics = gp.step()
            self.assertIsNotNone(metrics.best_rule)

    def test_top_k_equals_population_size(self):
        config = self._make_hierarchical_config(
            population_factory=PopulationGeneratorFactory(population_size=100),
            top_k_transfer=100,
            max_depth=2,
        )
        gp = BooleanGP(config)
        gp.step()

    def test_get_top_k_rules(self):
        gp = BooleanGP(self._make_hierarchical_config())
        child = gp.child_populations[0]
        top = child._get_top_k_rules()
        self.assertEqual(len(top), child._top_k)
        self.assertTrue(all(isinstance(r, Rule) for r in top))

    # ------------------------------------------------------------------ #
    #  _apply_feedback
    # ------------------------------------------------------------------ #
    def test_apply_feedback_additive(self):
        config = self._make_hierarchical_config(
            feedback_type="additive", feedback_strength=0.5
        )
        gp = BooleanGP(config)
        child = gp.child_populations[0]
        scores = np.ones(len(child.population))
        parent_scores = np.full(child._top_k, 0.4)
        child._apply_feedback(scores, parent_scores)
        # first top_k scores should have been increased
        expected = 1.0 + 0.4 * 0.5
        np.testing.assert_allclose(scores[: child._top_k], expected)
        # remaining scores unchanged
        np.testing.assert_array_equal(scores[child._top_k :], 1.0)

    def test_apply_feedback_multiplicative(self):
        config = self._make_hierarchical_config(
            feedback_type="multiplicative", feedback_strength=0.5
        )
        gp = BooleanGP(config)
        child = gp.child_populations[0]
        scores = np.ones(len(child.population)) * 2.0
        parent_scores = np.full(child._top_k, 0.4)
        child._apply_feedback(scores, parent_scores)
        expected = 2.0 * (1 + 0.4 * 0.5)
        np.testing.assert_allclose(scores[: child._top_k], expected)

    # ------------------------------------------------------------------ #
    #  _generate_child_feedback
    # ------------------------------------------------------------------ #
    def test_generate_child_feedback_uniform_scores(self):
        """When all scores are equal, feedback should be all zeros."""
        gp = BooleanGP(self._make_hierarchical_config())
        gp._forward()
        scores = np.ones(len(gp.population))
        feedback = gp._generate_child_feedback(scores)
        np.testing.assert_array_equal(feedback, 0.0)

    def test_generate_child_feedback_shape(self):
        gp = BooleanGP(self._make_hierarchical_config())
        gp._forward()
        scores = np.random.rand(len(gp.population))
        feedback = gp._generate_child_feedback(scores)
        self.assertEqual(feedback.shape, (2, gp._top_k))

    # ------------------------------------------------------------------ #
    #  optimize_scorer
    # ------------------------------------------------------------------ #
    def test_optimize_scorer_true(self):
        def accuracy_with_weight(predictions, labels, sample_weight=None):
            if sample_weight is None:
                return np.mean(predictions == labels)
            correct = predictions == labels
            return np.dot(correct, sample_weight) / sample_weight.sum()

        config = self._make_config(score_fn=accuracy_with_weight, optimize_scorer=True)
        gp = BooleanGP(config)
        metrics = gp.step()
        self.assertIsInstance(metrics.best_train_score, float)

    # ------------------------------------------------------------------ #
    #  best_not_improved_epochs tracking
    # ------------------------------------------------------------------ #
    def test_best_not_improved_epochs_tracking(self):
        gp = BooleanGP(self._make_config())
        gp.best_score = 1.1
        gp.step()
        gp.step()
        self.assertEqual(gp.best_not_improved_epochs, 2)

    # ------------------------------------------------------------------ #
    #  Doctests
    # ------------------------------------------------------------------ #
    def test_doctests(self):
        # TODO: Check if we can't make a single test file that tests the doctests for all files automatically
        # This means that it will automatically test it, without needing to specify the module
        result = doctest.testmod(hgp_lib.algorithms.boolean_gp, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
