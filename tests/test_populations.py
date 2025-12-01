import unittest
import random
import numpy as np

from hgp_lib.populations import PopulationGenerator, RandomStrategy, BestLiteralStrategy
from hgp_lib.rules import And, Or, Literal


class TestPopulations(unittest.TestCase):
    def setUp(self):
        # Seed random generators for reproducibility
        random.seed(42)
        np.random.seed(42)

        # Prepare dummy data
        self.train_data = np.array(
            [
                [True, False, True, False],
                [False, True, False, True],
                [True, True, False, False],
                [False, False, True, True],
            ]
        )
        self.train_labels = np.array([1, 0, 1, 0])
        self.num_literals = 4

    def simple_score_fn(self, predictions, labels):
        return np.mean(predictions == labels)

    def test_random_strategy_init(self):
        strategy = RandomStrategy(num_literals=10, operator_types=(And, Or))
        self.assertEqual(strategy.num_literals, 10)

        with self.assertRaises(ValueError):
            RandomStrategy(num_literals=0)

        with self.assertRaises(ValueError):
            RandomStrategy(num_literals=10, operator_types=[])

    def test_random_strategy_generate(self):
        strategy = RandomStrategy(
            num_literals=self.num_literals, operator_types=(And, Or)
        )
        rules = strategy.generate(n=1)
        self.assertEqual(len(rules), 1)
        rule = rules[0]

        self.assertIsInstance(rule, (And, Or))
        self.assertEqual(len(rule.subrules), 2)
        self.assertIsInstance(rule.subrules[0], Literal)
        self.assertIsInstance(rule.subrules[1], Literal)

        self.assertTrue(0 <= rule.subrules[0].value < self.num_literals)
        self.assertTrue(0 <= rule.subrules[1].value < self.num_literals)

    def test_random_strategy_generate_batch(self):
        strategy = RandomStrategy(
            num_literals=self.num_literals, operator_types=(And, Or)
        )
        rules = strategy.generate(n=10)
        self.assertEqual(len(rules), 10)
        for rule in rules:
            self.assertIsInstance(rule, (And, Or))
            self.assertEqual(len(rule.subrules), 2)

    def test_best_literal_strategy_init(self):
        BestLiteralStrategy(
            num_literals=self.train_data.shape[1],
            score_fn=self.simple_score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
        )

        with self.assertRaises(ValueError):
            BestLiteralStrategy(
                num_literals=self.num_literals,
                score_fn=self.simple_score_fn,
                train_data=self.train_data,
                train_labels=self.train_labels,
                sample_size=1.5,
            )

        with self.assertRaises(ValueError):
            BestLiteralStrategy(
                num_literals=self.num_literals,
                score_fn=self.simple_score_fn,
                train_data=self.train_data,
                train_labels=self.train_labels,
                feature_size=0,
            )

    def test_best_literal_strategy_generate(self):
        strategy = BestLiteralStrategy(
            num_literals=self.num_literals,
            score_fn=self.simple_score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            sample_size=None,
            feature_size=None,
        )

        rules = strategy.generate(n=1)
        rule = rules[0]

        self.assertIsInstance(rule, Literal)
        self.assertEqual(rule.value, 0)
        self.assertFalse(rule.negated)

        strategy_subset = BestLiteralStrategy(
            num_literals=self.num_literals,
            score_fn=self.simple_score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
            sample_size=2,
            feature_size=2,
        )
        rules_subset = strategy_subset.generate(n=1)
        rule_subset = rules_subset[0]
        self.assertIsInstance(rule_subset, Literal)

    def test_best_literal_strategy_no_labels(self):
        with self.assertRaises(ValueError):
            BestLiteralStrategy(
                num_literals=self.num_literals,
                score_fn=self.simple_score_fn,
                train_data=self.train_data,
                train_labels=None,
            )

    def test_best_literal_strategy_validation(self):
        with self.assertRaises(ValueError):
            BestLiteralStrategy(
                num_literals=10,
                score_fn=self.simple_score_fn,
                train_data=self.train_data,
                train_labels=self.train_labels[:-1],
            )

        with self.assertRaises(ValueError):
            BestLiteralStrategy(
                num_literals=10,
                score_fn=self.simple_score_fn,
                train_data=np.array([]),
                train_labels=np.array([]),
            )

        with self.assertRaises(ValueError):
            BestLiteralStrategy(
                num_literals=10,
                score_fn=self.simple_score_fn,
                train_data=self.train_data,
                train_labels=self.train_labels,
            )

        with self.assertRaises(ValueError):
            BestLiteralStrategy(
                num_literals=self.num_literals,
                score_fn=self.simple_score_fn,
                train_data=None,
                train_labels=self.train_labels,
            )

        with self.assertRaises(TypeError):
            BestLiteralStrategy(
                num_literals=self.num_literals,
                score_fn=self.simple_score_fn,
                train_data=[[1, 0]],
                train_labels=self.train_labels,
            )

        with self.assertRaises(TypeError):
            BestLiteralStrategy(
                num_literals=self.num_literals,
                score_fn=self.simple_score_fn,
                train_data=self.train_data,
                train_labels=[1, 0],
            )

    def test_population_generator_init(self):
        s1 = RandomStrategy(self.num_literals)

        PopulationGenerator(strategies=[s1], population_size=10)

        with self.assertRaises(ValueError):
            PopulationGenerator(strategies=[])

        with self.assertRaises(TypeError):
            PopulationGenerator(strategies=[s1, "not_a_strategy"])

        with self.assertRaises(ValueError):
            PopulationGenerator(strategies=[s1], weights=[-1.0])

    def test_population_generator_generate(self):
        s1 = RandomStrategy(self.num_literals)
        s2 = BestLiteralStrategy(
            num_literals=self.num_literals,
            score_fn=self.simple_score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
        )

        gen = PopulationGenerator(
            strategies=[s1, s2], population_size=50, weights=[0.5, 0.5]
        )
        population = gen.generate()

        self.assertEqual(len(population), 50)

        count_operators = sum(1 for r in population if isinstance(r, (And, Or)))
        count_literals = sum(1 for r in population if isinstance(r, Literal))

        self.assertGreater(count_operators, 0)
        self.assertGreater(count_literals, 0)

    def test_population_generator_weighted(self):
        s1 = RandomStrategy(self.num_literals)
        s2 = BestLiteralStrategy(
            num_literals=self.num_literals,
            score_fn=self.simple_score_fn,
            train_data=self.train_data,
            train_labels=self.train_labels,
        )

        gen_random = PopulationGenerator(
            strategies=[s1, s2], population_size=20, weights=[1.0, 0.0]
        )
        pop_random = gen_random.generate()
        self.assertTrue(all(isinstance(r, (And, Or)) for r in pop_random))

        gen_best = PopulationGenerator(
            strategies=[s1, s2], population_size=20, weights=[0.0, 1.0]
        )
        pop_best = gen_best.generate()
        self.assertTrue(all(isinstance(r, Literal) for r in pop_best))

    def test_doctests(self):
        import doctest
        import hgp_lib.populations.strategies
        import hgp_lib.populations.generator

        result = doctest.testmod(hgp_lib.populations.strategies, verbose=False)
        self.assertEqual(result.failed, 0, f"Strategies doctests failed: {result}")

        result = doctest.testmod(hgp_lib.populations.generator, verbose=False)
        self.assertEqual(result.failed, 0, f"Generator doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
