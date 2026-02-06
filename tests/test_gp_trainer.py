import doctest
import unittest

import numpy as np

import hgp_lib.trainers.gp_trainer
from hgp_lib.configs import BooleanGPConfig, TrainerConfig
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
        self.test_seed = 42

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

    def _make_gp_config(self, **kwargs):
        """Helper to create BooleanGPConfig with test defaults.

        Note: optimize_scorer=False by default since test scorer doesn't support sample_weight.
        """
        defaults = dict(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            optimize_scorer=False,
            seed=self.test_seed,
        )
        defaults.update(kwargs)
        return BooleanGPConfig(**defaults)

    def _make_trainer_config(self, gp_config=None, **kwargs):
        """Helper to create TrainerConfig with test defaults."""
        if gp_config is None:
            gp_config = self._make_gp_config()
        defaults = dict(
            gp_config=gp_config,
            num_epochs=10,
            progress_bar=False,
        )
        defaults.update(kwargs)
        return TrainerConfig(**defaults)

    def test_gp_trainer_validation(self):
        with self.subTest("score_fn must be callable"):
            with self.assertRaises(TypeError):
                gp_config = self._make_gp_config(score_fn="not callable")
                config = self._make_trainer_config(gp_config=gp_config)
                GPTrainer(config)

        with self.subTest("num_epochs must be int"):
            with self.assertRaises(TypeError):
                config = self._make_trainer_config(num_epochs=10.5)
                GPTrainer(config)

        with self.subTest("num_epochs must be positive"):
            with self.assertRaises(ValueError):
                config = self._make_trainer_config(num_epochs=0)
                GPTrainer(config)

        with self.subTest("train_data must be ndarray"):
            with self.assertRaises(TypeError):
                gp_config = self._make_gp_config(train_data="not array")
                config = self._make_trainer_config(gp_config=gp_config)
                GPTrainer(config)

        with self.subTest("train_labels must be ndarray"):
            with self.assertRaises(TypeError):
                gp_config = self._make_gp_config(train_labels="not array")
                config = self._make_trainer_config(gp_config=gp_config)
                GPTrainer(config)

        with self.subTest("train_labels length must match train_data rows"):
            with self.assertRaises(ValueError):
                gp_config = self._make_gp_config(train_labels=np.array([1, 0]))
                config = self._make_trainer_config(gp_config=gp_config)
                GPTrainer(config)

        with self.subTest("val_data and val_labels must both be provided or both None"):
            with self.assertRaises(ValueError):
                config = self._make_trainer_config(
                    val_data=self.val_data, val_labels=None
                )
                GPTrainer(config)

        with self.subTest("val_labels length must match val_data rows"):
            with self.assertRaises(ValueError):
                config = self._make_trainer_config(
                    val_data=self.val_data, val_labels=np.array([1])
                )
                GPTrainer(config)

        with self.subTest("val_score_fn must be callable if provided"):
            with self.assertRaises(TypeError):
                config = self._make_trainer_config(val_score_fn="not callable")
                GPTrainer(config)

        with self.subTest("val_every must be positive"):
            with self.assertRaises(ValueError):
                config = self._make_trainer_config(val_every=0)
                GPTrainer(config)

    def test_gp_trainer_init(self):
        config = self._make_trainer_config()
        trainer = GPTrainer(config)

        self.assertEqual(trainer.num_epochs, 10)
        self.assertIsNotNone(trainer.gp_algo)
        self.assertIsNone(trainer.val_data)
        self.assertIsNone(trainer.val_labels)

    def test_gp_trainer_init_with_validation(self):
        config = self._make_trainer_config(
            val_data=self.val_data, val_labels=self.val_labels
        )
        trainer = GPTrainer(config)

        self.assertIsNotNone(trainer.val_data)
        self.assertIsNotNone(trainer.val_labels)

    def test_gp_trainer_defaults(self):
        config = self._make_trainer_config()
        trainer = GPTrainer(config)

        self.assertIsInstance(trainer.gp_algo.crossover_executor, CrossoverExecutor)
        self.assertIsInstance(trainer.gp_algo.selection, TournamentSelection)
        self.assertEqual(trainer.val_every, 100)
        self.assertFalse(trainer.progress_bar)

    def test_fit_returns_trainer_result(self):
        config = self._make_trainer_config(num_epochs=5)
        trainer = GPTrainer(config)

        result = trainer.fit()

        self.assertIsNotNone(result.train_history)
        self.assertIsNotNone(result.best_rule)
        self.assertIsNotNone(result.best_score)
        self.assertEqual(len(result.train_history.epochs), 5)
        self.assertIsNone(result.val_history)

    def test_fit_with_validation(self):
        config = self._make_trainer_config(
            num_epochs=10,
            val_data=self.val_data,
            val_labels=self.val_labels,
            val_every=5,
        )
        trainer = GPTrainer(config)

        result = trainer.fit()

        self.assertEqual(len(result.train_history.epochs), 10)
        self.assertIsNotNone(result.val_history)
        self.assertEqual(len(result.val_history.epochs), 2)
        self.assertEqual(result.val_history.epochs[0].epoch, 4)
        self.assertEqual(result.val_history.epochs[1].epoch, 9)

    def test_score_returns_validate_best_metrics(self):
        config = self._make_trainer_config(num_epochs=5)
        trainer = GPTrainer(config)

        trainer.fit()
        test_metrics = trainer.score(self.test_data, self.test_labels)

        self.assertIsNotNone(test_metrics.best)
        self.assertIsNotNone(test_metrics.best_rule)
        self.assertIsInstance(test_metrics.best, float)
        self.assertIsInstance(test_metrics.best_rule, Rule)

    def test_score_validation(self):
        config = self._make_trainer_config(num_epochs=5)
        trainer = GPTrainer(config)

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
        config = self._make_trainer_config(num_epochs=5)
        trainer = GPTrainer(config)

        trainer.fit()

        def custom_score(predictions, labels):
            return np.sum(predictions & labels)

        test_metrics = trainer.score(
            self.test_data, self.test_labels, score_fn=custom_score
        )
        self.assertIsNotNone(test_metrics.best)

    def test_score_all_time_best(self):
        config = self._make_trainer_config(num_epochs=5)
        trainer = GPTrainer(config)

        trainer.fit()

        all_time_metrics = trainer.score(
            self.test_data, self.test_labels, all_time_best=True
        )
        current_metrics = trainer.score(
            self.test_data, self.test_labels, all_time_best=False
        )

        self.assertIsInstance(all_time_metrics.best_rule, Rule)
        self.assertIsInstance(current_metrics.best_rule, Rule)

    def test_custom_population_generator(self):
        generator = PopulationGenerator(
            strategies=[RandomStrategy(num_literals=self.num_features)],
            population_size=20,
        )

        gp_config = self._make_gp_config(population_generator=generator)
        config = self._make_trainer_config(gp_config=gp_config, num_epochs=5)
        trainer = GPTrainer(config)

        self.assertEqual(len(trainer.gp_algo.population), 20)

    def test_custom_mutation_executor(self):
        mutation_executor = MutationExecutor(
            literal_mutations=create_standard_literal_mutations(self.num_features),
            operator_mutations=create_standard_operator_mutations(self.num_features),
            mutation_p=0.5,
        )

        gp_config = self._make_gp_config(mutation_executor=mutation_executor)
        config = self._make_trainer_config(gp_config=gp_config, num_epochs=5)
        trainer = GPTrainer(config)

        self.assertEqual(trainer.gp_algo.mutation_executor.mutation_p, 0.5)

    def test_custom_crossover_executor(self):
        crossover_executor = CrossoverExecutor(crossover_p=0.5)

        gp_config = self._make_gp_config(crossover_executor=crossover_executor)
        config = self._make_trainer_config(gp_config=gp_config, num_epochs=5)
        trainer = GPTrainer(config)

        self.assertEqual(trainer.gp_algo.crossover_executor.crossover_p, 0.5)

    def test_custom_selection(self):
        selection = RouletteSelection()

        gp_config = self._make_gp_config(selection=selection)
        config = self._make_trainer_config(gp_config=gp_config, num_epochs=5)
        trainer = GPTrainer(config)

        self.assertIsInstance(trainer.gp_algo.selection, RouletteSelection)

    def test_regeneration(self):
        gp_config = self._make_gp_config(regeneration=True, regeneration_patience=50)
        config = self._make_trainer_config(gp_config=gp_config, num_epochs=5)
        trainer = GPTrainer(config)

        self.assertTrue(trainer.gp_algo.regeneration)
        self.assertEqual(trainer.gp_algo.regeneration_patience, 50)

    def test_progress_bar_disabled(self):
        config = self._make_trainer_config(num_epochs=5, progress_bar=False)
        trainer = GPTrainer(config)

        self.assertFalse(trainer.progress_bar)
        result = trainer.fit()
        self.assertEqual(len(result.train_history.epochs), 5)

    def test_val_score_fn_defaults_to_score_fn(self):
        config = self._make_trainer_config(num_epochs=5)
        trainer = GPTrainer(config)

        self.assertEqual(trainer.val_score_fn, trainer.score_fn)

    def test_train_history_tracks_best_scores(self):
        config = self._make_trainer_config(num_epochs=10)
        trainer = GPTrainer(config)

        result = trainer.fit()

        for epoch_metrics in result.train_history.epochs:
            self.assertIsInstance(epoch_metrics.best_score, float)
            self.assertGreaterEqual(epoch_metrics.best_score, 0.0)
            self.assertLessEqual(epoch_metrics.best_score, 1.0)

    def test_score_before_fit_raises_error(self):
        config = self._make_trainer_config(num_epochs=5)
        trainer = GPTrainer(config)

        # score() before fit() should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            trainer.score(self.test_data, self.test_labels)

        self.assertIn("No best rule available", str(context.exception))

    def test_optimize_scorer_true(self):
        """Test that optimize_scorer=True works with a scorer supporting sample_weight."""

        def accuracy_with_weight(predictions, labels, sample_weight=None):
            if sample_weight is None:
                return np.mean(predictions == labels)
            correct = predictions == labels
            return np.dot(correct, sample_weight) / sample_weight.sum()

        gp_config = self._make_gp_config(
            score_fn=accuracy_with_weight, optimize_scorer=True
        )
        config = self._make_trainer_config(gp_config=gp_config, num_epochs=5)
        trainer = GPTrainer(config)

        result = trainer.fit()
        self.assertIsNotNone(result.best_rule)
        self.assertIsInstance(result.best_score, float)

    def test_optimize_scorer_true_with_validation(self):
        """Test that optimize_scorer=True optimizes both train and validation scorers."""

        def accuracy_with_weight(predictions, labels, sample_weight=None):
            if sample_weight is None:
                return np.mean(predictions == labels)
            correct = predictions == labels
            return np.dot(correct, sample_weight) / sample_weight.sum()

        gp_config = self._make_gp_config(
            score_fn=accuracy_with_weight, optimize_scorer=True
        )
        config = self._make_trainer_config(
            gp_config=gp_config,
            num_epochs=10,
            val_data=self.val_data,
            val_labels=self.val_labels,
            val_every=5,
        )
        trainer = GPTrainer(config)

        result = trainer.fit()
        self.assertIsNotNone(result.val_history)
        self.assertEqual(len(result.val_history.epochs), 2)

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.trainers.gp_trainer, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
