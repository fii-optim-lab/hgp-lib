import doctest
import unittest
import random

import numpy as np

import hgp_lib.trainers.gp_trainer
from hgp_lib.configs import BooleanGPConfig, TrainerConfig
from hgp_lib.trainers import GPTrainer
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.metrics import PopulationHistory, GenerationMetrics
from hgp_lib.populations import PopulationGeneratorFactory
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

    def _make_gp_config(self, **kwargs):
        defaults = dict(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            optimize_scorer=False,
        )
        defaults.update(kwargs)
        return BooleanGPConfig(**defaults)

    def _make_trainer_config(self, gp_config=None, **kwargs):
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

    def test_fit_returns_population_history(self):
        config = self._make_trainer_config(num_epochs=5)
        trainer = GPTrainer(config)

        result = trainer.fit()

        self.assertIsInstance(result, PopulationHistory)
        self.assertEqual(len(result.generations), 5)

    def test_fit_generations_have_correct_structure(self):
        config = self._make_trainer_config(num_epochs=5)
        trainer = GPTrainer(config)

        result = trainer.fit()

        for i, gen in enumerate(result.generations):
            self.assertIsInstance(gen, GenerationMetrics)
            self.assertIsNotNone(gen.best_rule)
            self.assertIsNotNone(gen.train_scores)
            self.assertGreater(len(gen.train_scores), 0)

    def test_fit_with_validation(self):
        config = self._make_trainer_config(
            num_epochs=10,
            val_data=self.val_data,
            val_labels=self.val_labels,
            val_every=5,
        )
        trainer = GPTrainer(config)

        result = trainer.fit()

        self.assertEqual(len(result.generations), 10)

        for i, gen in enumerate(result.generations):
            if (i + 1) % 5 == 0:
                self.assertIsNotNone(
                    gen.val_score, f"Generation {i} should have val_score"
                )
            else:
                self.assertIsNone(
                    gen.val_score, f"Generation {i} should not have val_score"
                )

    def test_custom_population_factory(self):
        factory = PopulationGeneratorFactory(population_size=20)

        gp_config = self._make_gp_config(population_factory=factory)
        config = self._make_trainer_config(gp_config=gp_config, num_epochs=5)
        trainer = GPTrainer(config)

        self.assertEqual(len(trainer.gp_algo.population), 20)

    def test_custom_mutation_factory(self):
        from hgp_lib.mutations import MutationExecutorFactory

        factory = MutationExecutorFactory(mutation_p=0.5)

        gp_config = self._make_gp_config(mutation_factory=factory)
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
        self.assertEqual(len(result.generations), 5)

    def test_val_score_fn_defaults_to_score_fn(self):
        config = self._make_trainer_config(num_epochs=5)
        trainer = GPTrainer(config)

        self.assertEqual(trainer.val_score_fn, trainer.config.gp_config.score_fn)

    def test_train_history_tracks_best_scores(self):
        config = self._make_trainer_config(num_epochs=10)
        trainer = GPTrainer(config)

        result = trainer.fit()

        for gen in result.generations:
            self.assertIsInstance(gen.best_train_score, float)
            self.assertGreaterEqual(gen.best_train_score, 0.0)
            self.assertLessEqual(gen.best_train_score, 1.0)

    def test_optimize_scorer_true(self):
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
        self.assertEqual(len(result.generations), 5)

    def test_optimize_scorer_true_with_validation(self):
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
        self.assertIsNotNone(result.generations[4].val_score)
        self.assertIsNotNone(result.generations[9].val_score)

    def test_best_val_score_from_history(self):
        config = self._make_trainer_config(
            num_epochs=10,
            val_data=self.val_data,
            val_labels=self.val_labels,
            val_every=5,
        )
        trainer = GPTrainer(config)

        result = trainer.fit()

        best_val = result.best_val_score
        self.assertIsNotNone(best_val)
        self.assertIsInstance(best_val, float)

    def test_global_best_rule_from_history(self):
        config = self._make_trainer_config(
            num_epochs=10,
            val_data=self.val_data,
            val_labels=self.val_labels,
            val_every=5,
        )
        trainer = GPTrainer(config)

        result = trainer.fit()
        self.assertIsInstance(result.global_best_rule, Rule)

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.trainers.gp_trainer, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
