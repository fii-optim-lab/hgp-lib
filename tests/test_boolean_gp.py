import doctest
import unittest
import random

import numpy as np

import hgp_lib.algorithms.boolean_gp
from hgp_lib.algorithms import BooleanGP
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.mutations import (
    MutationExecutor,
    create_standard_literal_mutations,
    create_standard_operator_mutations,
)
from hgp_lib.populations import PopulationGenerator, RandomStrategy
from hgp_lib.rules import Rule


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

    def test_boolean_gp_validation(self):
        with self.subTest("score_fn must be callable"):
            with self.assertRaises(TypeError):
                BooleanGP(
                    score_fn="not callable",
                    population_generator=self.generator,
                    mutation_executor=self.mutation_executor,
                )

        with self.subTest("population_generator must be PopulationGenerator"):
            with self.assertRaises(TypeError):
                BooleanGP(
                    score_fn=self.score_fn,
                    population_generator="not generator",
                    mutation_executor=self.mutation_executor,
                )

        with self.subTest("mutation_executor must be MutationExecutor"):
            with self.assertRaises(TypeError):
                BooleanGP(
                    score_fn=self.score_fn,
                    population_generator=self.generator,
                    mutation_executor="not executor",
                )

        with self.subTest("crossover_executor must be CrossoverExecutor"):
            with self.assertRaises(TypeError):
                BooleanGP(
                    score_fn=self.score_fn,
                    population_generator=self.generator,
                    mutation_executor=self.mutation_executor,
                    crossover_executor="not executor",
                )

        with self.subTest("selection must be BaseSelection"):
            with self.assertRaises(TypeError):
                BooleanGP(
                    score_fn=self.score_fn,
                    population_generator=self.generator,
                    mutation_executor=self.mutation_executor,
                    selection="not selection",
                )

        with self.subTest("regeneration must be bool"):
            with self.assertRaises(TypeError):
                BooleanGP(
                    score_fn=self.score_fn,
                    population_generator=self.generator,
                    mutation_executor=self.mutation_executor,
                    regeneration=1,
                )

        with self.subTest("regeneration_patience must be int"):
            with self.assertRaises(TypeError):
                BooleanGP(
                    score_fn=self.score_fn,
                    population_generator=self.generator,
                    mutation_executor=self.mutation_executor,
                    regeneration_patience=1.5,
                )

        with self.subTest(
            "regeneration_patience must be positive when regeneration=True"
        ):
            with self.assertRaises(ValueError):
                BooleanGP(
                    score_fn=self.score_fn,
                    population_generator=self.generator,
                    mutation_executor=self.mutation_executor,
                    regeneration=True,
                    regeneration_patience=0,
                )

    def test_boolean_gp_init(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        self.assertIsNotNone(gp.population)
        self.assertEqual(len(gp.population), 10)
        self.assertEqual(gp.best_score, -float("inf"))
        self.assertIsNone(gp.best_rule)
        self.assertEqual(gp.real_best_score, -float("inf"))
        self.assertIsNone(gp.real_best_rule)
        self.assertEqual(gp.best_not_improved_epochs, 0)

    def test_boolean_gp_defaults(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        self.assertIsInstance(gp.crossover_executor, CrossoverExecutor)
        from hgp_lib.selections import RouletteSelection

        self.assertIsInstance(gp.selection, RouletteSelection)

    def test_step_returns_metrics(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        metrics = gp.step(self.train_data, self.train_labels)

        self.assertIn("best", metrics)
        self.assertIn("best_rule", metrics)
        self.assertIn("real_best", metrics)
        self.assertIn("real_best_rule", metrics)
        self.assertIn("current_best", metrics)
        self.assertIn("population_scores", metrics)
        self.assertIn("epoch", metrics)
        self.assertIn("best_not_improved_epochs", metrics)
        self.assertIn("regenerated", metrics)

        self.assertEqual(metrics["epoch"], 0)
        self.assertIsInstance(metrics["best_rule"], Rule)
        self.assertIsInstance(metrics["real_best_rule"], Rule)
        self.assertIsInstance(metrics["population_scores"], np.ndarray)
        self.assertGreater(len(metrics["population_scores"]), 0)

    def test_step_updates_best_rule(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        initial_best = gp.best_score
        metrics = gp.step(self.train_data, self.train_labels)

        self.assertGreaterEqual(metrics["best"], initial_best)
        self.assertIsNotNone(gp.best_rule)
        self.assertIsNotNone(gp.real_best_rule)

    def test_step_increments_epoch(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        metrics1 = gp.step(self.train_data, self.train_labels)
        self.assertEqual(metrics1["epoch"], 0)

        metrics2 = gp.step(self.train_data, self.train_labels)
        self.assertEqual(metrics2["epoch"], 1)

    def test_step_updates_population_size(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        initial_size = len(gp.population)
        gp.step(self.train_data, self.train_labels)

        self.assertEqual(len(gp.population), initial_size)

    def test_validate_best_raises_without_steps(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        with self.assertRaises(RuntimeError) as context:
            gp.validate_best(self.val_data, self.val_labels)

        self.assertIn("No best rule available", str(context.exception))

    def test_validate_best_returns_metrics(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step(self.train_data, self.train_labels)
        metrics = gp.validate_best(self.val_data, self.val_labels)

        self.assertIn("best", metrics)
        self.assertIn("best_rule", metrics)
        self.assertIsInstance(metrics["best"], float)
        self.assertIsInstance(metrics["best_rule"], Rule)

    def test_validate_best_all_time_best(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step(self.train_data, self.train_labels)
        current_metrics = gp.validate_best(
            self.val_data, self.val_labels, all_time_best=False
        )
        all_time_metrics = gp.validate_best(
            self.val_data, self.val_labels, all_time_best=True
        )

        self.assertIsInstance(current_metrics["best_rule"], Rule)
        self.assertIsInstance(all_time_metrics["best_rule"], Rule)

    def test_validate_best_custom_score_fn(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step(self.train_data, self.train_labels)

        def custom_score(predictions, labels):
            return np.sum(predictions & labels)

        metrics = gp.validate_best(
            self.val_data, self.val_labels, score_fn=custom_score
        )
        self.assertIn("best", metrics)

    def test_validate_population_raises_without_steps(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        with self.assertRaises(RuntimeError) as context:
            gp.validate_population(self.val_data, self.val_labels)

        self.assertIn("No best rule available", str(context.exception))

    def test_validate_population_returns_metrics(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step(self.train_data, self.train_labels)
        metrics = gp.validate_population(self.val_data, self.val_labels)

        self.assertIn("best", metrics)
        self.assertIn("best_rule", metrics)
        self.assertIn("population_scores", metrics)
        self.assertIsInstance(metrics["best"], float)
        self.assertIsInstance(metrics["best_rule"], Rule)
        self.assertEqual(len(metrics["population_scores"]), len(gp.population))

    def test_validate_population_all_time_best(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step(self.train_data, self.train_labels)
        current_metrics = gp.validate_population(
            self.val_data, self.val_labels, all_time_best=False
        )
        all_time_metrics = gp.validate_population(
            self.val_data, self.val_labels, all_time_best=True
        )

        self.assertIsInstance(current_metrics["best_rule"], Rule)
        self.assertIsInstance(all_time_metrics["best_rule"], Rule)

    def test_validate_population_custom_score_fn(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step(self.train_data, self.train_labels)

        def custom_score(predictions, labels):
            return np.sum(predictions & labels)

        metrics = gp.validate_population(
            self.val_data, self.val_labels, score_fn=custom_score
        )
        self.assertIn("best", metrics)
        self.assertIn("population_scores", metrics)

    def test_regeneration_disabled(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
            regeneration=False,
            regeneration_patience=1,
        )

        gp.step(self.train_data, self.train_labels)

        self.assertFalse(gp.regeneration)

    def test_regeneration_enabled(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
            regeneration=True,
            regeneration_patience=1,
        )

        gp.step(self.train_data, self.train_labels)

        metrics = gp.step(self.train_data, self.train_labels)

        if metrics["regenerated"]:
            self.assertEqual(gp.best_score, -float("inf"))
            self.assertEqual(gp.best_not_improved_epochs, 0)

    def test_best_not_improved_epochs_tracking(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        metrics1 = gp.step(self.train_data, self.train_labels)
        initial_epochs = metrics1["best_not_improved_epochs"]

        metrics2 = gp.step(self.train_data, self.train_labels)

        if metrics2["current_best"] < metrics1["best"]:
            self.assertGreater(metrics2["best_not_improved_epochs"], initial_epochs)
        else:
            self.assertEqual(metrics2["best_not_improved_epochs"], 0)

    def test_real_best_tracking(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step(self.train_data, self.train_labels)
        initial_real_best = gp.real_best_score

        gp.step(self.train_data, self.train_labels)

        self.assertGreaterEqual(gp.real_best_score, initial_real_best)
        if gp.best_score > initial_real_best:
            self.assertEqual(gp.real_best_score, gp.best_score)

    def test_multiple_steps(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        for i in range(5):
            metrics = gp.step(self.train_data, self.train_labels)
            self.assertEqual(metrics["epoch"], i)
            self.assertIsNotNone(metrics["best_rule"])

    def test_step_with_custom_crossover(self):
        crossover_executor = CrossoverExecutor(crossover_p=0.5)
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
            crossover_executor=crossover_executor,
        )

        metrics = gp.step(self.train_data, self.train_labels)
        self.assertIn("best", metrics)

    def test_step_with_custom_selection(self):
        from hgp_lib.selections import RouletteSelection

        selection = RouletteSelection()
        gp = BooleanGP(
            score_fn=self.score_fn,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
            selection=selection,
        )

        metrics = gp.step(self.train_data, self.train_labels)
        self.assertIn("best", metrics)

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.algorithms.boolean_gp, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
