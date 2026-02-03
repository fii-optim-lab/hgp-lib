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
from hgp_lib.populations.sampling import (
    FeatureSamplingStrategy,
    InstanceSamplingStrategy,
    CombinedSamplingStrategy,
    SamplingStrategy,
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
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                    population_generator=self.generator,
                    mutation_executor=self.mutation_executor,
                )

        with self.subTest("population_generator must be PopulationGenerator"):
            with self.assertRaises(TypeError):
                BooleanGP(
                    score_fn=self.score_fn,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                    population_generator="not generator",
                    mutation_executor=self.mutation_executor,
                )

        with self.subTest("mutation_executor must be MutationExecutor"):
            with self.assertRaises(TypeError):
                BooleanGP(
                    score_fn=self.score_fn,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                    population_generator=self.generator,
                    mutation_executor="not executor",
                )

        with self.subTest("crossover_executor must be CrossoverExecutor"):
            with self.assertRaises(TypeError):
                BooleanGP(
                    score_fn=self.score_fn,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                    population_generator=self.generator,
                    mutation_executor=self.mutation_executor,
                    crossover_executor="not executor",
                )

        with self.subTest("selection must be BaseSelection"):
            with self.assertRaises(TypeError):
                BooleanGP(
                    score_fn=self.score_fn,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                    population_generator=self.generator,
                    mutation_executor=self.mutation_executor,
                    selection="not selection",
                )

        with self.subTest("regeneration must be bool"):
            with self.assertRaises(TypeError):
                BooleanGP(
                    score_fn=self.score_fn,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                    population_generator=self.generator,
                    mutation_executor=self.mutation_executor,
                    regeneration=1,
                )

        with self.subTest("regeneration_patience must be int"):
            with self.assertRaises(TypeError):
                BooleanGP(
                    score_fn=self.score_fn,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
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
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                    population_generator=self.generator,
                    mutation_executor=self.mutation_executor,
                    regeneration=True,
                    regeneration_patience=0,
                )

    def test_boolean_gp_init(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
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
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        self.assertIsInstance(gp.crossover_executor, CrossoverExecutor)
        from hgp_lib.selections import RouletteSelection

        self.assertIsInstance(gp.selection, RouletteSelection)

    def test_step_returns_metrics(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        metrics = gp.step()

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
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        initial_best = gp.best_score
        metrics = gp.step()

        self.assertGreaterEqual(metrics["best"], initial_best)
        self.assertIsNotNone(gp.best_rule)
        self.assertIsNotNone(gp.real_best_rule)

    def test_step_increments_epoch(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        metrics1 = gp.step()
        self.assertEqual(metrics1["epoch"], 0)

        metrics2 = gp.step()
        self.assertEqual(metrics2["epoch"], 1)

    def test_step_updates_population_size(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        initial_size = len(gp.population)
        gp.step()

        self.assertEqual(len(gp.population), initial_size)

    def test_validate_best_raises_without_steps(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        with self.assertRaises(RuntimeError) as context:
            gp.validate_best(self.val_data, self.val_labels)

        self.assertIn("No best rule available", str(context.exception))

    def test_validate_best_returns_metrics(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step()
        metrics = gp.validate_best(self.val_data, self.val_labels)

        self.assertIn("best", metrics)
        self.assertIn("best_rule", metrics)
        self.assertIsInstance(metrics["best"], float)
        self.assertIsInstance(metrics["best_rule"], Rule)

    def test_validate_best_all_time_best(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step()
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
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step()

        def custom_score(predictions, labels):
            return np.sum(predictions & labels)

        metrics = gp.validate_best(
            self.val_data, self.val_labels, score_fn=custom_score
        )
        self.assertIn("best", metrics)

    def test_validate_population_raises_without_steps(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        with self.assertRaises(RuntimeError) as context:
            gp.validate_population(self.val_data, self.val_labels)

        self.assertIn("No best rule available", str(context.exception))

    def test_validate_population_returns_metrics(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step()
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
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step()
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
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step()

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
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
            regeneration=False,
            regeneration_patience=1,
        )

        gp.step()

        self.assertFalse(gp.regeneration)

    def test_regeneration_enabled(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
            regeneration=True,
            regeneration_patience=1,
        )

        gp.step()
        metrics = gp.step()

        if metrics["regenerated"]:
            self.assertEqual(gp.best_score, -float("inf"))
            self.assertEqual(gp.best_not_improved_epochs, 0)

    def test_best_not_improved_epochs_tracking(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        metrics1 = gp.step()
        initial_epochs = metrics1["best_not_improved_epochs"]

        metrics2 = gp.step()

        if metrics2["current_best"] < metrics1["best"]:
            self.assertGreater(metrics2["best_not_improved_epochs"], initial_epochs)
        else:
            self.assertEqual(metrics2["best_not_improved_epochs"], 0)

    def test_real_best_tracking(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        gp.step()
        initial_real_best = gp.real_best_score

        gp.step()

        self.assertGreaterEqual(gp.real_best_score, initial_real_best)
        if gp.best_score > initial_real_best:
            self.assertEqual(gp.real_best_score, gp.best_score)

    def test_multiple_steps(self):
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
        )

        for i in range(5):
            metrics = gp.step()
            self.assertEqual(metrics["epoch"], i)
            self.assertIsNotNone(metrics["best_rule"])

    def test_step_with_custom_crossover(self):
        crossover_executor = CrossoverExecutor(crossover_p=0.5)
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
            crossover_executor=crossover_executor,
        )

        metrics = gp.step()
        self.assertIn("best", metrics)

    def test_step_with_custom_selection(self):
        from hgp_lib.selections import RouletteSelection

        selection = RouletteSelection()
        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
            selection=selection,
        )

        metrics = gp.step()
        self.assertIn("best", metrics)

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.algorithms.boolean_gp, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


class TestBooleanGPHierarchicalParameterValidation(unittest.TestCase):
    """Property tests for BooleanGP hierarchical parameter validation.

    **Property 10: BooleanGP Parameter Validation**
    **Validates: Requirements 6.4, 6.5, 6.6, 7.6**

    Tests that:
    - Negative num_child_populations SHALL raise ValueError
    - Negative depth SHALL raise ValueError
    - Non-SamplingStrategy sampling_strategy SHALL raise TypeError
    - depth > 0 with sampling_strategy=None SHALL raise ValueError
    """

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

    def test_negative_num_child_populations_raises_value_error(self):
        """Negative num_child_populations SHALL raise ValueError.

        **Property 10: BooleanGP Parameter Validation**
        **Validates: Requirements 6.4**
        """
        negative_values = [-1, -5, -100]

        for value in negative_values:
            with self.subTest(num_child_populations=value):
                with self.assertRaises(ValueError) as context:
                    BooleanGP(
                        score_fn=self.score_fn,
                        train_data=self.train_data,
                        train_labels=self.train_labels,
                        population_generator=self.generator,
                        mutation_executor=self.mutation_executor,
                        num_child_populations=value,
                    )
                self.assertIn(
                    "num_child_populations must be non-negative", str(context.exception)
                )

    def test_negative_depth_raises_value_error(self):
        """Negative depth SHALL raise ValueError.

        **Property 10: BooleanGP Parameter Validation**
        **Validates: Requirements 6.5**
        """
        negative_values = [-1, -5, -100]

        for value in negative_values:
            with self.subTest(depth=value):
                with self.assertRaises(ValueError) as context:
                    BooleanGP(
                        score_fn=self.score_fn,
                        train_data=self.train_data,
                        train_labels=self.train_labels,
                        population_generator=self.generator,
                        mutation_executor=self.mutation_executor,
                        depth=value,
                    )
                self.assertIn("depth must be non-negative", str(context.exception))

    def test_non_sampling_strategy_raises_type_error(self):
        """Non-SamplingStrategy sampling_strategy SHALL raise TypeError.

        **Property 10: BooleanGP Parameter Validation**
        **Validates: Requirements 6.6**
        """
        invalid_strategies = ["not a strategy", 123, [], {}, object()]

        for invalid_strategy in invalid_strategies:
            with self.subTest(sampling_strategy=type(invalid_strategy).__name__):
                with self.assertRaises(TypeError):
                    BooleanGP(
                        score_fn=self.score_fn,
                        train_data=self.train_data,
                        train_labels=self.train_labels,
                        population_generator=self.generator,
                        mutation_executor=self.mutation_executor,
                        sampling_strategy=invalid_strategy,
                    )

    def test_depth_gt_0_without_sampling_strategy_raises_value_error(self):
        """depth > 0 with sampling_strategy=None SHALL raise ValueError.

        **Property 10: BooleanGP Parameter Validation**
        **Validates: Requirements 7.6**
        """
        positive_depths = [1, 2, 5, 10]

        for depth in positive_depths:
            with self.subTest(depth=depth):
                with self.assertRaises(ValueError) as context:
                    BooleanGP(
                        score_fn=self.score_fn,
                        train_data=self.train_data,
                        train_labels=self.train_labels,
                        population_generator=self.generator,
                        mutation_executor=self.mutation_executor,
                        depth=depth,
                        sampling_strategy=None,
                    )
                self.assertIn(
                    "sampling_strategy required when depth > 0", str(context.exception)
                )

    def test_valid_hierarchical_parameters_with_depth_0(self):
        """Valid parameters with depth=0 should not raise any errors.

        When depth=0, no sampling_strategy is required.
        """
        valid_num_child_populations = [0, 1, 5, 10]

        for num_child_populations in valid_num_child_populations:
            with self.subTest(num_child_populations=num_child_populations):
                gp = BooleanGP(
                    score_fn=self.score_fn,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                    population_generator=self.generator,
                    mutation_executor=self.mutation_executor,
                    num_child_populations=num_child_populations,
                    depth=0,
                    sampling_strategy=None,
                )
                self.assertEqual(gp.num_child_populations, num_child_populations)
                self.assertEqual(gp.depth, 0)
                self.assertIsNone(gp.sampling_strategy)

    def test_valid_hierarchical_parameters_with_sampling_strategy(self):
        """Valid parameters with sampling_strategy should not raise any errors."""
        from hgp_lib.populations.sampling import FeatureSamplingStrategy

        strategy = FeatureSamplingStrategy(feature_fraction=1.0)

        gp = BooleanGP(
            score_fn=self.score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            population_generator=self.generator,
            mutation_executor=self.mutation_executor,
            num_child_populations=3,
            depth=0,
            sampling_strategy=strategy,
        )

        self.assertEqual(gp.num_child_populations, 3)
        self.assertEqual(gp.depth, 0)
        self.assertIs(gp.sampling_strategy, strategy)

    def test_num_child_populations_must_be_int(self):
        """num_child_populations must be an integer.

        **Property 10: BooleanGP Parameter Validation**
        **Validates: Requirements 6.4**
        """
        invalid_types = [1.5, "3", [3], None]

        for value in invalid_types:
            with self.subTest(num_child_populations=type(value).__name__):
                with self.assertRaises(TypeError):
                    BooleanGP(
                        score_fn=self.score_fn,
                        train_data=self.train_data,
                        train_labels=self.train_labels,
                        population_generator=self.generator,
                        mutation_executor=self.mutation_executor,
                        num_child_populations=value,
                    )

    def test_depth_must_be_int(self):
        """depth must be an integer.

        **Property 10: BooleanGP Parameter Validation**
        **Validates: Requirements 6.5**
        """
        invalid_types = [1.5, "2", [2], None]

        for value in invalid_types:
            with self.subTest(depth=type(value).__name__):
                with self.assertRaises(TypeError):
                    BooleanGP(
                        score_fn=self.score_fn,
                        train_data=self.train_data,
                        train_labels=self.train_labels,
                        population_generator=self.generator,
                        mutation_executor=self.mutation_executor,
                        depth=value,
                    )


if __name__ == "__main__":
    unittest.main()


class TestHierarchicalStructureFormation(unittest.TestCase):
    """Tests for hierarchical structure formation.

    Tests that:
    - depth=0 → len(child_populations) == 0
    - depth>0 with strategy → len(child_populations) == num_child_populations
    - Each child has depth = parent.depth - 1
    """

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

        def accuracy(predictions, labels):
            return np.mean(predictions == labels)

        self.score_fn = accuracy

    def _create_gp(
        self,
        num_features: int,
        num_instances: int,
        num_child_populations: int,
        depth: int,
        sampling_strategy: SamplingStrategy | None,
    ) -> BooleanGP:
        """Helper to create BooleanGP with given parameters."""
        train_data = np.random.rand(num_instances, num_features) > 0.5
        train_labels = np.random.randint(0, 2, num_instances)

        generator = PopulationGenerator(
            strategies=[RandomStrategy(num_literals=num_features)],
            population_size=5,
        )

        mutation_executor = MutationExecutor(
            literal_mutations=create_standard_literal_mutations(num_features),
            operator_mutations=create_standard_operator_mutations(num_features),
        )

        return BooleanGP(
            score_fn=self.score_fn,
            train_data=train_data,
            train_labels=train_labels,
            population_generator=generator,
            mutation_executor=mutation_executor,
            num_child_populations=num_child_populations,
            depth=depth,
            sampling_strategy=sampling_strategy,
        )

    def test_depth_0_no_children(self):
        """depth=0 → len(child_populations) == 0 regardless of num_child_populations."""
        test_cases = [
            {"num_features": 4, "num_instances": 10, "num_child_populations": 0},
            {"num_features": 8, "num_instances": 20, "num_child_populations": 3},
            {"num_features": 10, "num_instances": 50, "num_child_populations": 5},
        ]

        for params in test_cases:
            with self.subTest(**params):
                gp = self._create_gp(
                    num_features=params["num_features"],
                    num_instances=params["num_instances"],
                    num_child_populations=params["num_child_populations"],
                    depth=0,
                    sampling_strategy=None,
                )
                self.assertEqual(len(gp.child_populations), 0)

    def test_depth_gt_0_creates_children(self):
        """depth>0 with strategy → len(child_populations) == num_child_populations."""
        test_cases = [
            {
                "num_features": 8,
                "num_instances": 20,
                "num_child_populations": 2,
                "depth": 1,
            },
            {
                "num_features": 10,
                "num_instances": 30,
                "num_child_populations": 3,
                "depth": 1,
            },
            {
                "num_features": 12,
                "num_instances": 40,
                "num_child_populations": 4,
                "depth": 2,
            },
        ]

        for params in test_cases:
            with self.subTest(**params):
                strategy = FeatureSamplingStrategy(feature_fraction=1.0)
                gp = self._create_gp(
                    num_features=params["num_features"],
                    num_instances=params["num_instances"],
                    num_child_populations=params["num_child_populations"],
                    depth=params["depth"],
                    sampling_strategy=strategy,
                )
                self.assertEqual(
                    len(gp.child_populations), params["num_child_populations"]
                )

    def test_child_depth_decremented(self):
        """Each child has depth = parent.depth - 1."""
        test_cases = [
            {
                "num_features": 8,
                "num_instances": 20,
                "num_child_populations": 2,
                "depth": 1,
            },
            {
                "num_features": 10,
                "num_instances": 30,
                "num_child_populations": 3,
                "depth": 2,
            },
        ]

        for params in test_cases:
            with self.subTest(**params):
                strategy = FeatureSamplingStrategy(feature_fraction=1.0)
                gp = self._create_gp(
                    num_features=params["num_features"],
                    num_instances=params["num_instances"],
                    num_child_populations=params["num_child_populations"],
                    depth=params["depth"],
                    sampling_strategy=strategy,
                )
                for child in gp.child_populations:
                    self.assertEqual(child.depth, params["depth"] - 1)

    def test_recursive_hierarchy_depth_2(self):
        """depth=2 creates children with grandchildren."""
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        gp = self._create_gp(
            num_features=12,
            num_instances=40,
            num_child_populations=2,
            depth=2,
            sampling_strategy=strategy,
        )

        # Parent has 2 children
        self.assertEqual(len(gp.child_populations), 2)

        # Each child has 2 grandchildren
        for child in gp.child_populations:
            self.assertEqual(child.depth, 1)
            self.assertEqual(len(child.child_populations), 2)

            # Grandchildren have depth 0 and no children
            for grandchild in child.child_populations:
                self.assertEqual(grandchild.depth, 0)
                self.assertEqual(len(grandchild.child_populations), 0)


class TestChildConfigurationCorrectness(unittest.TestCase):
    """Tests for child configuration correctness.

    Tests that:
    - child.train_data.shape[1] == len(child.feature_mapping)
    - child.feature_mapping maps local indices 0..n-1 to valid parent indices
    - child.num_child_populations == parent.num_child_populations
    - child.sampling_strategy == parent.sampling_strategy
    """

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

        def accuracy(predictions, labels):
            return np.mean(predictions == labels)

        self.score_fn = accuracy

    def _create_gp(
        self,
        num_features: int,
        num_instances: int,
        num_child_populations: int,
        depth: int,
        sampling_strategy: SamplingStrategy,
    ) -> BooleanGP:
        """Helper to create BooleanGP with given parameters."""
        train_data = np.random.rand(num_instances, num_features) > 0.5
        train_labels = np.random.randint(0, 2, num_instances)

        generator = PopulationGenerator(
            strategies=[RandomStrategy(num_literals=num_features)],
            population_size=5,
        )

        mutation_executor = MutationExecutor(
            literal_mutations=create_standard_literal_mutations(num_features),
            operator_mutations=create_standard_operator_mutations(num_features),
        )

        return BooleanGP(
            score_fn=self.score_fn,
            train_data=train_data,
            train_labels=train_labels,
            population_generator=generator,
            mutation_executor=mutation_executor,
            num_child_populations=num_child_populations,
            depth=depth,
            sampling_strategy=sampling_strategy,
        )

    def test_feature_mapping_size_matches_data(self):
        """child.train_data.shape[1] == len(child.feature_mapping)."""
        test_cases = [
            {"num_features": 8, "num_instances": 20, "num_child_populations": 2},
            {"num_features": 10, "num_instances": 30, "num_child_populations": 3},
            {"num_features": 16, "num_instances": 50, "num_child_populations": 4},
        ]

        for params in test_cases:
            with self.subTest(**params):
                strategy = FeatureSamplingStrategy(feature_fraction=1.0)
                gp = self._create_gp(
                    num_features=params["num_features"],
                    num_instances=params["num_instances"],
                    num_child_populations=params["num_child_populations"],
                    depth=1,
                    sampling_strategy=strategy,
                )

                for child in gp.child_populations:
                    self.assertIsNotNone(child.feature_mapping)
                    self.assertEqual(
                        child.train_data.shape[1],
                        len(child.feature_mapping),
                    )

    def test_feature_mapping_valid_indices(self):
        """child.feature_mapping maps local indices 0..n-1 to valid parent indices."""
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        num_features = 10
        gp = self._create_gp(
            num_features=num_features,
            num_instances=30,
            num_child_populations=3,
            depth=1,
            sampling_strategy=strategy,
        )

        for child in gp.child_populations:
            self.assertIsNotNone(child.feature_mapping)
            num_child_features = len(child.feature_mapping)

            # Check local indices are 0..n-1
            expected_local_indices = set(range(num_child_features))
            actual_local_indices = set(child.feature_mapping.keys())
            self.assertEqual(expected_local_indices, actual_local_indices)

            # Check parent indices are valid
            for parent_idx in child.feature_mapping.values():
                self.assertGreaterEqual(parent_idx, 0)
                self.assertLess(parent_idx, num_features)

    def test_child_inherits_num_child_populations(self):
        """child.num_child_populations == parent.num_child_populations."""
        test_cases = [2, 3, 4]

        for num_child_populations in test_cases:
            with self.subTest(num_child_populations=num_child_populations):
                strategy = FeatureSamplingStrategy(feature_fraction=1.0)
                gp = self._create_gp(
                    num_features=10,
                    num_instances=30,
                    num_child_populations=num_child_populations,
                    depth=1,
                    sampling_strategy=strategy,
                )

                for child in gp.child_populations:
                    self.assertEqual(child.num_child_populations, num_child_populations)

    def test_child_inherits_sampling_strategy(self):
        """child.sampling_strategy == parent.sampling_strategy."""
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        gp = self._create_gp(
            num_features=10,
            num_instances=30,
            num_child_populations=3,
            depth=1,
            sampling_strategy=strategy,
        )

        for child in gp.child_populations:
            self.assertIs(child.sampling_strategy, strategy)

    def test_child_has_correct_score_fn(self):
        """child.score_fn == parent.score_fn."""
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        gp = self._create_gp(
            num_features=10,
            num_instances=30,
            num_child_populations=3,
            depth=1,
            sampling_strategy=strategy,
        )

        for child in gp.child_populations:
            self.assertIs(child.score_fn, gp.score_fn)


class TestFeatureMappingRoundTrip(unittest.TestCase):
    """Tests for feature mapping round-trip.

    Tests that:
    - For any feature_indices array [a, b, c, ...], the constructed feature_mapping
      {0: a, 1: b, 2: c, ...} correctly translates local indices to parent indices.
    """

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

        def accuracy(predictions, labels):
            return np.mean(predictions == labels)

        self.score_fn = accuracy

    def _create_gp(
        self,
        num_features: int,
        num_instances: int,
        num_child_populations: int,
        sampling_strategy: SamplingStrategy,
    ) -> BooleanGP:
        """Helper to create BooleanGP with given parameters."""
        train_data = np.random.rand(num_instances, num_features) > 0.5
        train_labels = np.random.randint(0, 2, num_instances)

        generator = PopulationGenerator(
            strategies=[RandomStrategy(num_literals=num_features)],
            population_size=5,
        )

        mutation_executor = MutationExecutor(
            literal_mutations=create_standard_literal_mutations(num_features),
            operator_mutations=create_standard_operator_mutations(num_features),
        )

        return BooleanGP(
            score_fn=self.score_fn,
            train_data=train_data,
            train_labels=train_labels,
            population_generator=generator,
            mutation_executor=mutation_executor,
            num_child_populations=num_child_populations,
            depth=1,
            sampling_strategy=sampling_strategy,
        )

    def test_feature_mapping_translates_correctly(self):
        """Feature mapping correctly translates local indices to parent indices.

        For any child, accessing child.train_data[:, local_idx] should give the same
        data as parent.train_data[:, feature_mapping[local_idx]] for the corresponding
        instances.
        """
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        gp = self._create_gp(
            num_features=10,
            num_instances=30,
            num_child_populations=3,
            sampling_strategy=strategy,
        )

        for child in gp.child_populations:
            self.assertIsNotNone(child.feature_mapping)

            # For feature sampling, all instances are preserved
            # So child.train_data[:, local_idx] should equal
            # parent.train_data[:, feature_mapping[local_idx]]
            for local_idx, parent_idx in child.feature_mapping.items():
                child_column = child.train_data[:, local_idx]
                parent_column = gp.train_data[:, parent_idx]
                np.testing.assert_array_equal(
                    child_column,
                    parent_column,
                    err_msg=f"Feature mapping failed: local {local_idx} -> parent {parent_idx}",
                )

    def test_instance_sampling_preserves_all_features(self):
        """Instance sampling preserves all features in feature_mapping."""
        num_features = 10
        strategy = InstanceSamplingStrategy(instance_fraction=1.0)
        gp = self._create_gp(
            num_features=num_features,
            num_instances=30,
            num_child_populations=3,
            sampling_strategy=strategy,
        )

        for child in gp.child_populations:
            self.assertIsNotNone(child.feature_mapping)

            # Instance sampling should preserve all features
            self.assertEqual(len(child.feature_mapping), num_features)

            # Feature mapping should be identity: {0: 0, 1: 1, ..., n-1: n-1}
            for local_idx in range(num_features):
                self.assertEqual(child.feature_mapping[local_idx], local_idx)

    def test_combined_sampling_feature_mapping(self):
        """Combined sampling creates valid feature mapping."""
        num_features = 10
        strategy = CombinedSamplingStrategy(
            feature_fraction=1.0,
            instance_fraction=1.0,
        )
        gp = self._create_gp(
            num_features=num_features,
            num_instances=30,
            num_child_populations=3,
            sampling_strategy=strategy,
        )

        for child in gp.child_populations:
            self.assertIsNotNone(child.feature_mapping)

            # Feature mapping should have correct size
            self.assertEqual(
                len(child.feature_mapping),
                child.train_data.shape[1],
            )

            # All parent indices should be valid
            for parent_idx in child.feature_mapping.values():
                self.assertGreaterEqual(parent_idx, 0)
                self.assertLess(parent_idx, num_features)

    def test_feature_sampling_with_high_fraction(self):
        """Feature sampling with fraction > 1.0 uses replacement."""
        strategy = FeatureSamplingStrategy(feature_fraction=1.5)
        gp = self._create_gp(
            num_features=10,
            num_instances=30,
            num_child_populations=3,
            sampling_strategy=strategy,
        )

        for child in gp.child_populations:
            self.assertIsNotNone(child.feature_mapping)
            # With replacement, we may have more features than base_count
            self.assertGreater(len(child.feature_mapping), 0)

    def test_child_data_shape_consistency(self):
        """Child data shape is consistent with feature mapping."""
        test_cases = [
            {"strategy": FeatureSamplingStrategy(feature_fraction=0.5)},
            {"strategy": FeatureSamplingStrategy(feature_fraction=1.0)},
            {"strategy": InstanceSamplingStrategy(instance_fraction=1.0)},
            {
                "strategy": CombinedSamplingStrategy(
                    feature_fraction=1.0, instance_fraction=1.0
                )
            },
        ]

        for params in test_cases:
            with self.subTest(strategy=type(params["strategy"]).__name__):
                gp = self._create_gp(
                    num_features=10,
                    num_instances=30,
                    num_child_populations=3,
                    sampling_strategy=params["strategy"],
                )

                for child in gp.child_populations:
                    # Data columns match feature mapping size
                    self.assertEqual(
                        child.train_data.shape[1],
                        len(child.feature_mapping),
                    )
                    # Labels match data rows
                    self.assertEqual(
                        len(child.train_labels),
                        child.train_data.shape[0],
                    )
