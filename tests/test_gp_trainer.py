import doctest
import unittest
import random

import numpy as np

import hgp_lib.trainers.gp_trainer
from hgp_lib.trainers import GPTrainer
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.mutations import (
    MutationExecutor,
    create_standard_literal_mutations,
    create_standard_operator_mutations,
)
from hgp_lib.populations import PopulationGenerator, RandomStrategy
from hgp_lib.selections import RouletteSelection, TournamentSelection
from hgp_lib.rules import Rule


class TestGPTrainer(unittest.TestCase):
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
        self.test_data = np.array([[True, True, True, False]])
        self.test_labels = np.array([1])
        self.num_features = 4

        def accuracy(predictions, labels):
            return np.mean(predictions == labels)

        self.score_fn = accuracy

    def test_gp_trainer_validation(self):
        with self.subTest("score_fn must be callable"):
            with self.assertRaises(TypeError):
                GPTrainer(
                    score_fn="not callable",
                    num_epochs=10,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                )

        with self.subTest("num_epochs must be int"):
            with self.assertRaises(TypeError):
                GPTrainer(
                    score_fn=self.score_fn,
                    num_epochs=10.5,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                )

        with self.subTest("num_epochs must be positive"):
            with self.assertRaises(ValueError):
                GPTrainer(
                    score_fn=self.score_fn,
                    num_epochs=0,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                )

        with self.subTest("train_data must be ndarray"):
            with self.assertRaises(TypeError):
                GPTrainer(
                    score_fn=self.score_fn,
                    num_epochs=10,
                    train_data="not array",
                    train_labels=self.train_labels,
                )

        with self.subTest("train_labels must be ndarray"):
            with self.assertRaises(TypeError):
                GPTrainer(
                    score_fn=self.score_fn,
                    num_epochs=10,
                    train_data=self.train_data,
                    train_labels="not array",
                )

        with self.subTest("train_labels length must match train_data rows"):
            with self.assertRaises(ValueError):
                GPTrainer(
                    score_fn=self.score_fn,
                    num_epochs=10,
                    train_data=self.train_data,
                    train_labels=np.array([1, 0]),
                )

        with self.subTest("val_data and val_labels must both be provided or both None"):
            with self.assertRaises(ValueError):
                GPTrainer(
                    score_fn=self.score_fn,
                    num_epochs=10,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                    val_data=self.val_data,
                    val_labels=None,
                )

        with self.subTest("val_labels length must match val_data rows"):
            with self.assertRaises(ValueError):
                GPTrainer(
                    score_fn=self.score_fn,
                    num_epochs=10,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                    val_data=self.val_data,
                    val_labels=np.array([1]),
                )

        with self.subTest("val_score_fn must be callable if provided"):
            with self.assertRaises(TypeError):
                GPTrainer(
                    score_fn=self.score_fn,
                    num_epochs=10,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                    val_score_fn="not callable",
                )

        with self.subTest("val_every must be positive"):
            with self.assertRaises(ValueError):
                GPTrainer(
                    score_fn=self.score_fn,
                    num_epochs=10,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                    val_every=0,
                )

    def test_gp_trainer_init(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=10,
            train_data=self.train_data,
            train_labels=self.train_labels,
            progress_bar=False,
        )

        self.assertEqual(trainer.num_epochs, 10)
        self.assertIsNotNone(trainer.gp_algo)
        self.assertIsNone(trainer.val_data)
        self.assertIsNone(trainer.val_labels)

    def test_gp_trainer_init_with_validation(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=10,
            train_data=self.train_data,
            train_labels=self.train_labels,
            val_data=self.val_data,
            val_labels=self.val_labels,
            progress_bar=False,
        )

        self.assertIsNotNone(trainer.val_data)
        self.assertIsNotNone(trainer.val_labels)

    def test_gp_trainer_defaults(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=10,
            train_data=self.train_data,
            train_labels=self.train_labels,
            progress_bar=False,
        )

        self.assertIsInstance(trainer.gp_algo.crossover_executor, CrossoverExecutor)
        self.assertIsInstance(trainer.gp_algo.selection, TournamentSelection)
        self.assertEqual(trainer.val_every, 100)
        self.assertFalse(trainer.progress_bar)

    def test_fit_returns_trainer_metrics(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=5,
            train_data=self.train_data,
            train_labels=self.train_labels,
            progress_bar=False,
        )

        metrics = trainer.fit()

        self.assertIn("train_best_history", metrics)
        self.assertIn("val_best_history", metrics)
        self.assertIn("val_epochs", metrics)
        self.assertEqual(len(metrics["train_best_history"]), 5)
        self.assertEqual(len(metrics["val_best_history"]), 0)
        self.assertEqual(len(metrics["val_epochs"]), 0)

    def test_fit_with_validation(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=10,
            train_data=self.train_data,
            train_labels=self.train_labels,
            val_data=self.val_data,
            val_labels=self.val_labels,
            val_every=5,
            progress_bar=False,
        )

        metrics = trainer.fit()

        self.assertEqual(len(metrics["train_best_history"]), 10)
        self.assertEqual(len(metrics["val_best_history"]), 2)
        self.assertEqual(len(metrics["val_epochs"]), 2)
        self.assertEqual(metrics["val_epochs"][0], 4)
        self.assertEqual(metrics["val_epochs"][1], 9)

    def test_fit_with_val_score_fn(self):
        def custom_val_score(predictions, labels):
            return np.sum(predictions & labels)

        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=10,
            train_data=self.train_data,
            train_labels=self.train_labels,
            val_data=self.val_data,
            val_labels=self.val_labels,
            val_score_fn=custom_val_score,
            val_every=5,
            progress_bar=False,
        )

        metrics = trainer.fit()
        self.assertEqual(len(metrics["val_best_history"]), 2)

    def test_score_returns_validate_best_metrics(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=5,
            train_data=self.train_data,
            train_labels=self.train_labels,
            progress_bar=False,
        )

        trainer.fit()
        test_metrics = trainer.score(self.test_data, self.test_labels)

        self.assertIn("best", test_metrics)
        self.assertIn("best_rule", test_metrics)
        self.assertIsInstance(test_metrics["best"], float)
        self.assertIsInstance(test_metrics["best_rule"], Rule)

    def test_score_validation(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=5,
            train_data=self.train_data,
            train_labels=self.train_labels,
            progress_bar=False,
        )

        trainer.fit()

        with self.subTest("test_data must be ndarray"):
            with self.assertRaises(TypeError):
                trainer.score("not array", self.test_labels)

        with self.subTest("test_labels must be ndarray"):
            with self.assertRaises(TypeError):
                trainer.score(self.test_data, "not array")

        with self.subTest("test_labels length must match test_data rows"):
            with self.assertRaises(ValueError):
                trainer.score(self.test_data, np.array([1, 0]))

    def test_score_with_custom_score_fn(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=5,
            train_data=self.train_data,
            train_labels=self.train_labels,
            progress_bar=False,
        )

        trainer.fit()

        def custom_score(predictions, labels):
            return np.sum(predictions & labels)

        test_metrics = trainer.score(
            self.test_data, self.test_labels, score_fn=custom_score
        )
        self.assertIn("best", test_metrics)

    def test_score_all_time_best(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=5,
            train_data=self.train_data,
            train_labels=self.train_labels,
            progress_bar=False,
        )

        trainer.fit()

        all_time_metrics = trainer.score(
            self.test_data, self.test_labels, all_time_best=True
        )
        current_metrics = trainer.score(
            self.test_data, self.test_labels, all_time_best=False
        )

        self.assertIsInstance(all_time_metrics["best_rule"], Rule)
        self.assertIsInstance(current_metrics["best_rule"], Rule)

    def test_custom_population_generator(self):
        generator = PopulationGenerator(
            strategies=[RandomStrategy(num_literals=self.num_features)],
            population_size=20,
        )

        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=5,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=generator,
            progress_bar=False,
        )

        self.assertEqual(len(trainer.gp_algo.population), 20)

    def test_custom_mutation_executor(self):
        mutation_executor = MutationExecutor(
            literal_mutations=create_standard_literal_mutations(self.num_features),
            operator_mutations=create_standard_operator_mutations(self.num_features),
            mutation_p=0.5,
        )

        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=5,
            train_data=self.train_data,
            train_labels=self.train_labels,
            mutation_executor=mutation_executor,
            progress_bar=False,
        )

        self.assertEqual(trainer.gp_algo.mutation_executor.mutation_p, 0.5)

    def test_custom_crossover_executor(self):
        crossover_executor = CrossoverExecutor(crossover_p=0.5)

        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=5,
            train_data=self.train_data,
            train_labels=self.train_labels,
            crossover_executor=crossover_executor,
            progress_bar=False,
        )

        self.assertEqual(trainer.gp_algo.crossover_executor.crossover_p, 0.5)

    def test_custom_selection(self):
        selection = RouletteSelection()

        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=5,
            train_data=self.train_data,
            train_labels=self.train_labels,
            selection=selection,
            progress_bar=False,
        )

        self.assertIsInstance(trainer.gp_algo.selection, RouletteSelection)

    def test_regeneration(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=5,
            train_data=self.train_data,
            train_labels=self.train_labels,
            regeneration=True,
            regeneration_patience=50,
            progress_bar=False,
        )

        self.assertTrue(trainer.gp_algo.regeneration)
        self.assertEqual(trainer.gp_algo.regeneration_patience, 50)

    def test_progress_bar_disabled(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=5,
            train_data=self.train_data,
            train_labels=self.train_labels,
            progress_bar=False,
        )

        self.assertFalse(trainer.progress_bar)
        metrics = trainer.fit()
        self.assertEqual(len(metrics["train_best_history"]), 5)

    def test_val_score_fn_defaults_to_score_fn(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=5,
            train_data=self.train_data,
            train_labels=self.train_labels,
            progress_bar=False,
        )

        self.assertEqual(trainer.val_score_fn, trainer.score_fn)

    def test_train_best_history_tracks_current_best(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=10,
            train_data=self.train_data,
            train_labels=self.train_labels,
            progress_bar=False,
        )

        metrics = trainer.fit()

        for score in metrics["train_best_history"]:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_score_before_fit_raises_error(self):
        trainer = GPTrainer(
            score_fn=self.score_fn,
            num_epochs=5,
            train_data=self.train_data,
            train_labels=self.train_labels,
            progress_bar=False,
        )

        # score() before fit() should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            trainer.score(self.test_data, self.test_labels)

        self.assertIn("No best rule available", str(context.exception))

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.trainers.gp_trainer, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
