"""Tests for ILPStrategy (hgp_lib.populations.strategies.ILPStrategy)."""

import unittest
import numpy as np

from hgp_lib.populations.strategies import ILPStrategy
from hgp_lib.populations import PopulationGenerator
from hgp_lib.populations.strategies import RandomStrategy
from hgp_lib.rules import And, Or, Literal, Rule


class TestILPStrategyValidation(unittest.TestCase):
    """Input validation tests for ILPStrategy.__init__."""

    def setUp(self):
        np.random.seed(42)
        self.data = np.random.rand(40, 6) > 0.5
        self.labels = np.random.randint(0, 2, 40)

    def test_invalid_operator_type_raises(self):
        with self.assertRaises(ValueError):
            ILPStrategy(
                num_literals=6, train_data=self.data, train_labels=self.labels,
                operator_type="xor",
            )

    def test_min_literals_below_2_raises(self):
        with self.assertRaises(ValueError):
            ILPStrategy(
                num_literals=6, train_data=self.data, train_labels=self.labels,
                min_literals=1,
            )

    def test_max_below_min_raises(self):
        with self.assertRaises(ValueError):
            ILPStrategy(
                num_literals=6, train_data=self.data, train_labels=self.labels,
                min_literals=4, max_literals=3,
            )

    def test_mismatched_num_literals_raises(self):
        with self.assertRaises(ValueError):
            ILPStrategy(
                num_literals=99, train_data=self.data, train_labels=self.labels,
            )

    def test_invalid_sample_size_float_raises(self):
        with self.assertRaises(ValueError):
            ILPStrategy(
                num_literals=6, train_data=self.data, train_labels=self.labels,
                sample_size=0.0,
            )

    def test_invalid_feature_size_int_raises(self):
        with self.assertRaises(ValueError):
            ILPStrategy(
                num_literals=6, train_data=self.data, train_labels=self.labels,
                feature_size=0,
            )


class TestILPStrategyGenerate(unittest.TestCase):
    """Tests for ILPStrategy.generate."""

    def setUp(self):
        np.random.seed(42)
        self.data = np.random.rand(60, 8) > 0.5
        self.labels = (self.data[:, 0] & self.data[:, 2]).astype(int)

    def _make_strategy(self, **kwargs):
        defaults = dict(
            num_literals=8,
            train_data=self.data,
            train_labels=self.labels,
            sample_size=30,
            feature_size=5,
            max_literals=4,
            min_literals=2,
            time_limit=5.0,
        )
        defaults.update(kwargs)
        return ILPStrategy(**defaults)

    def test_returns_correct_count(self):
        """generate(n) returns exactly n rules."""
        strategy = self._make_strategy(operator_type="and")
        rules = strategy.generate(n=3)
        self.assertEqual(len(rules), 3)

    def test_returns_empty_for_zero(self):
        """generate(0) returns an empty list."""
        strategy = self._make_strategy()
        self.assertEqual(strategy.generate(0), [])

    def test_returns_empty_for_negative(self):
        """generate(-1) returns an empty list."""
        strategy = self._make_strategy()
        self.assertEqual(strategy.generate(-1), [])

    def test_all_rules_are_rule_instances(self):
        """Every generated rule is a Rule instance."""
        strategy = self._make_strategy(operator_type="random")
        rules = strategy.generate(n=4)
        for r in rules:
            self.assertIsInstance(r, Rule)

    def test_and_operator_type_produces_and_rules(self):
        """operator_type='and' produces only And rules."""
        strategy = self._make_strategy(operator_type="and")
        rules = strategy.generate(n=5)
        for r in rules:
            self.assertIsInstance(r, And)

    def test_or_operator_type_produces_or_rules(self):
        """operator_type='or' produces only Or rules."""
        strategy = self._make_strategy(operator_type="or")
        rules = strategy.generate(n=5)
        for r in rules:
            self.assertIsInstance(r, Or)

    def test_rules_have_at_least_min_literals(self):
        """Each rule has at least min_literals subrules."""
        strategy = self._make_strategy(min_literals=2, operator_type="and")
        rules = strategy.generate(n=3)
        for r in rules:
            self.assertGreaterEqual(len(r.subrules), 2)

    def test_rules_have_at_most_max_literals(self):
        """Each rule has at most max_literals subrules."""
        strategy = self._make_strategy(max_literals=3, operator_type="or")
        rules = strategy.generate(n=3)
        for r in rules:
            self.assertLessEqual(len(r.subrules), 3)

    def test_subrules_are_literals(self):
        """All subrules of generated rules are Literal instances."""
        strategy = self._make_strategy(operator_type="and")
        rules = strategy.generate(n=3)
        for r in rules:
            for sub in r.subrules:
                self.assertIsInstance(sub, Literal)

    def test_literal_values_in_valid_range(self):
        """All literal values are in [0, num_literals)."""
        strategy = self._make_strategy()
        rules = strategy.generate(n=5)
        for r in rules:
            for sub in r.subrules:
                self.assertGreaterEqual(sub.value, 0)
                self.assertLess(sub.value, 8)

    def test_rules_are_evaluable(self):
        """Generated rules can be evaluated on the training data without error."""
        strategy = self._make_strategy()
        rules = strategy.generate(n=3)
        for r in rules:
            preds = r.evaluate(self.data)
            self.assertEqual(len(preds), len(self.data))

    def test_sample_size_as_float(self):
        """sample_size as float fraction works."""
        strategy = self._make_strategy(sample_size=0.5)
        rules = strategy.generate(n=2)
        self.assertEqual(len(rules), 2)

    def test_feature_size_as_float(self):
        """feature_size as float fraction works."""
        strategy = self._make_strategy(feature_size=0.5)
        rules = strategy.generate(n=2)
        self.assertEqual(len(rules), 2)

    def test_none_sample_and_feature_size(self):
        """sample_size=None and feature_size=None uses all data."""
        strategy = self._make_strategy(sample_size=None, feature_size=None)
        rules = strategy.generate(n=2)
        self.assertEqual(len(rules), 2)


class TestILPStrategyIntegration(unittest.TestCase):
    """Integration tests: ILPStrategy with PopulationGenerator."""

    def setUp(self):
        np.random.seed(42)
        self.data = np.random.rand(60, 8) > 0.5
        self.labels = np.random.randint(0, 2, 60)

    def test_mixed_strategies_with_weights(self):
        """PopulationGenerator correctly mixes RandomStrategy and ILPStrategy."""
        strategies = [
            RandomStrategy(num_literals=8),
            ILPStrategy(
                num_literals=8,
                train_data=self.data,
                train_labels=self.labels,
                sample_size=30,
                feature_size=5,
            ),
        ]
        gen = PopulationGenerator(
            strategies=strategies, population_size=10, weights=[0.7, 0.3]
        )
        pop = gen.generate()
        self.assertEqual(len(pop), 10)
        for r in pop:
            self.assertIsInstance(r, Rule)


if __name__ == "__main__":
    unittest.main()
