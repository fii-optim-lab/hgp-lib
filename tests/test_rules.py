import doctest
import unittest
import numpy as np

import hgp_lib
import hgp_lib.rules.utils
from hgp_lib.rules import Literal
from hgp_lib.rules.operators import And, Or
from hgp_lib.rules.low_memory_operators import And as LowMemoryAnd, Or as LowMemoryOr
from hgp_lib.rules.utils import replace_with_rule, deep_swap


class TestRules(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(10, 20) < 0.5

    def test_literal(self):
        np.testing.assert_array_equal(
            Literal(value=0, negated=True).evaluate(self.data), ~self.data[:, 0]
        )
        np.testing.assert_array_equal(
            Literal(value=0, negated=False).evaluate(self.data), self.data[:, 0]
        )
        np.testing.assert_array_equal(
            Literal(value=1, negated=False).evaluate(self.data), self.data[:, 1]
        )
        np.testing.assert_array_equal(
            Literal(value=1, negated=True).evaluate(self.data), ~self.data[:, 1]
        )

    def test_and(self):
        for and_type in [And, LowMemoryAnd]:
            with self.subTest(f"Testing {type(and_type()).__qualname__}"):
                same_rule = and_type(
                    subrules=[
                        Literal(value=0),
                        Literal(value=0),
                    ],
                    negated=False,
                )
                np.testing.assert_array_equal(
                    same_rule.evaluate(self.data), self.data[:, 0]
                )

                same_rule_negated = and_type(
                    subrules=[
                        Literal(value=1),
                        Literal(value=1),
                    ],
                    negated=True,
                )
                np.testing.assert_array_equal(
                    same_rule_negated.evaluate(self.data), ~self.data[:, 1]
                )

                all_zero = and_type(
                    subrules=[
                        Literal(value=2),
                        Literal(value=2, negated=True),
                    ],
                    negated=False,
                )
                np.testing.assert_array_equal(
                    all_zero.evaluate(self.data),
                    np.zeros(len(self.data), dtype=np.bool),
                )

                all_one = and_type(
                    subrules=[
                        Literal(value=3),
                        Literal(value=3, negated=True),
                    ],
                    negated=True,
                )
                np.testing.assert_array_equal(
                    all_one.evaluate(self.data), np.ones(len(self.data), dtype=np.bool)
                )

                test_rule_1 = and_type(
                    subrules=[
                        Literal(value=0),
                        Literal(value=1, negated=True),
                        Literal(value=2),
                    ],
                    negated=False,
                )
                result = self.data[:, 0] & ~self.data[:, 1] & self.data[:, 2]
                np.testing.assert_array_equal(test_rule_1.evaluate(self.data), result)

                test_rule_2 = and_type(
                    subrules=[
                        Literal(value=0),
                        Literal(value=1, negated=True),
                        Literal(value=4, negated=True),
                    ],
                    negated=True,
                )
                result = ~(self.data[:, 0] & ~self.data[:, 1] & ~self.data[:, 4])
                np.testing.assert_array_equal(test_rule_2.evaluate(self.data), result)

    def test_or(self):
        for or_type in [Or, LowMemoryOr]:
            with self.subTest(f"Testing {type(or_type()).__qualname__}"):
                same_rule = or_type(
                    subrules=[
                        Literal(value=0),
                        Literal(value=0),
                    ],
                    negated=False,
                )
                np.testing.assert_array_equal(
                    same_rule.evaluate(self.data), self.data[:, 0]
                )

                same_rule_negated = or_type(
                    subrules=[
                        Literal(value=1),
                        Literal(value=1),
                    ],
                    negated=True,
                )
                np.testing.assert_array_equal(
                    same_rule_negated.evaluate(self.data), ~self.data[:, 1]
                )

                all_one = or_type(
                    subrules=[
                        Literal(value=2),
                        Literal(value=2, negated=True),
                    ],
                    negated=False,
                )
                np.testing.assert_array_equal(
                    all_one.evaluate(self.data), np.ones(len(self.data), dtype=np.bool)
                )

                all_zero = or_type(
                    subrules=[
                        Literal(value=3),
                        Literal(value=3, negated=True),
                    ],
                    negated=True,
                )
                np.testing.assert_array_equal(
                    all_zero.evaluate(self.data),
                    np.zeros(len(self.data), dtype=np.bool),
                )

                test_rule_1 = or_type(
                    subrules=[
                        Literal(value=0),
                        Literal(value=1, negated=True),
                        Literal(value=2),
                    ],
                    negated=False,
                )
                result = self.data[:, 0] | ~self.data[:, 1] | self.data[:, 2]
                np.testing.assert_array_equal(test_rule_1.evaluate(self.data), result)

                test_rule_2 = or_type(
                    subrules=[
                        Literal(value=0),
                        Literal(value=1, negated=True),
                        Literal(value=4, negated=True),
                    ],
                    negated=True,
                )
                result = ~(self.data[:, 0] | ~self.data[:, 1] | ~self.data[:, 4])
                np.testing.assert_array_equal(test_rule_2.evaluate(self.data), result)

    def test_operators(self):
        test_rule_1 = Or(
            subrules=[
                And(
                    subrules=[
                        Literal(value=1),
                        Literal(value=2, negated=True),
                    ]
                ),
                And(
                    subrules=[
                        Or(
                            subrules=[
                                Literal(value=3),
                                Literal(value=4),
                                Literal(value=5, negated=True),
                            ]
                        ),
                        Literal(value=6, negated=True),
                    ]
                ),
                Literal(value=7),
                Literal(value=8, negated=True),
            ]
        )
        result = (
            self.data[:, 1] & ~self.data[:, 2]
            | (self.data[:, 3] | self.data[:, 4] | ~self.data[:, 5]) & ~self.data[:, 6]
            | self.data[:, 7]
            | ~self.data[:, 8]
        )
        np.testing.assert_array_equal(test_rule_1.evaluate(self.data), result)

    def test_replace_with_rule(self):
        """Test that replace_with_rule replaces target content in-place."""
        with self.subTest("And to Or"):
            target = And([Literal(value=0), Literal(value=1)])
            source = Or([Literal(value=2), Literal(value=3)])
            original_id = id(target)

            replace_with_rule(target, source)

            # Same object identity
            self.assertEqual(id(target), original_id)
            # Class changed
            self.assertEqual(type(target), Or)
            # Content changed
            self.assertEqual(str(target), "Or(2, 3)")
            # Subrules have correct parent
            for subrule in target.subrules:
                self.assertEqual(subrule.parent, target)

        with self.subTest("Or to And with negation"):
            target = Or([Literal(value=0)], negated=False)
            source = And([Literal(value=1), Literal(value=2)], negated=True)

            replace_with_rule(target, source)

            self.assertEqual(type(target), And)
            self.assertTrue(target.negated)
            self.assertEqual(str(target), "~And(1, 2)")

        with self.subTest("Operator to Literal"):
            target = And([Literal(value=0), Literal(value=1)])
            source = Literal(value=5, negated=True)

            replace_with_rule(target, source)

            self.assertEqual(type(target), Literal)
            self.assertEqual(target.value, 5)
            self.assertTrue(target.negated)
            self.assertEqual(str(target), "~5")

        with self.subTest("Literal to Operator"):
            target = Literal(value=0)
            source = Or([Literal(value=1), Literal(value=2)])

            replace_with_rule(target, source)

            self.assertEqual(type(target), Or)
            self.assertIsNone(target.value)
            self.assertEqual(len(target.subrules), 2)

        with self.subTest("Source unchanged"):
            target = And([Literal(value=0)])
            source = Or([Literal(value=1), Literal(value=2)])
            source_str = str(source)

            replace_with_rule(target, source)

            self.assertEqual(str(source), source_str)

    def test_deep_swap(self):
        """Test that deep_swap exchanges content between two nodes."""
        with self.subTest("Swap And and Or"):
            node_a = And([Literal(value=0), Literal(value=1)])
            node_b = Or([Literal(value=2), Literal(value=3)])
            id_a, id_b = id(node_a), id(node_b)

            deep_swap(node_a, node_b)

            # Same object identities
            self.assertEqual(id(node_a), id_a)
            self.assertEqual(id(node_b), id_b)
            # Content swapped
            self.assertEqual(type(node_a), Or)
            self.assertEqual(type(node_b), And)
            self.assertEqual(str(node_a), "Or(2, 3)")
            self.assertEqual(str(node_b), "And(0, 1)")

        with self.subTest("Swap with negation"):
            node_a = And([Literal(value=0)], negated=True)
            node_b = Or([Literal(value=1)], negated=False)

            deep_swap(node_a, node_b)

            self.assertFalse(node_a.negated)
            self.assertTrue(node_b.negated)

        with self.subTest("Swap Literal and Operator"):
            node_a = Literal(value=5, negated=True)
            node_b = And([Literal(value=0), Literal(value=1)])

            deep_swap(node_a, node_b)

            self.assertEqual(type(node_a), And)
            self.assertEqual(str(node_a), "And(0, 1)")
            self.assertEqual(type(node_b), Literal)
            self.assertEqual(node_b.value, 5)
            self.assertTrue(node_b.negated)

        with self.subTest("Swap two Literals"):
            node_a = Literal(value=0, negated=False)
            node_b = Literal(value=1, negated=True)

            deep_swap(node_a, node_b)

            self.assertEqual(node_a.value, 1)
            self.assertTrue(node_a.negated)
            self.assertEqual(node_b.value, 0)
            self.assertFalse(node_b.negated)

        with self.subTest("Subrules have correct parent after swap"):
            node_a = And([Literal(value=0), Literal(value=1)])
            node_b = Or([Literal(value=2), Literal(value=3)])

            deep_swap(node_a, node_b)

            for subrule in node_a.subrules:
                self.assertEqual(subrule.parent, node_a)
            for subrule in node_b.subrules:
                self.assertEqual(subrule.parent, node_b)

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.rules.rules, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.rules.literals, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.rules.operators, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.rules.low_memory_operators, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.rules.utils, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")

    def test_literal_evaluate_multiclass(self):
        data = np.array(
            [
                [True, False, True],
                [False, True, False],
                [True, True, False],
                [False, False, False],
            ]
        )
        lit = Literal(value=0, class_label=2)
        out = lit.evaluate_multiclass(data)
        np.testing.assert_array_equal(out, np.array([2, -1, 2, -1]))
        lit_neg = Literal(value=1, negated=True, class_label=1)
        out_neg = lit_neg.evaluate_multiclass(data)
        np.testing.assert_array_equal(out_neg, np.array([1, -1, -1, 1]))
        lit_none = Literal(value=2)
        out_none = lit_none.evaluate_multiclass(data)
        np.testing.assert_array_equal(out_none, np.array([-1, -1, -1, -1]))

    def test_and_evaluate_multiclass(self):
        data = np.array(
            [
                [True, False],
                [True, True],
                [False, True],
                [True, True],
            ]
        )
        rule = And(
            [
                Literal(value=0, class_label=3),
                Literal(value=1, class_label=3),
            ],
            class_label=3,
        )
        out = rule.evaluate_multiclass(data)
        np.testing.assert_array_equal(out, np.array([-1, 3, -1, 3]))
        rule_neg = And(
            [
                Literal(value=0, class_label=2),
                Literal(value=1, class_label=2),
            ],
            class_label=2,
            negated=True,
        )
        out_neg = rule_neg.evaluate_multiclass(data)
        np.testing.assert_array_equal(out_neg, np.array([2, -1, 2, -1]))

    def test_or_evaluate_multiclass(self):
        data = np.array(
            [
                [True, False],
                [False, True],
                [False, False],
                [True, True],
            ]
        )
        rule = Or(
            [
                Literal(value=0, class_label=1),
                Literal(value=1, class_label=1),
            ],
            class_label=1,
        )
        out = rule.evaluate_multiclass(data)
        np.testing.assert_array_equal(out, np.array([1, 1, -1, 1]))
        rule_neg = Or(
            [
                Literal(value=0, class_label=2),
                Literal(value=1, class_label=2),
            ],
            class_label=2,
            negated=True,
        )
        out_neg = rule_neg.evaluate_multiclass(data)
        np.testing.assert_array_equal(out_neg, np.array([-1, -1, 2, -1]))

    def test_ruleset_predict_multiclass(self):
        data = np.array(
            [
                [True, False],
                [False, True],
                [True, True],
                [False, False],
            ]
        )
        rule1 = Literal(value=0, class_label=1)
        rule2 = Literal(value=1, class_label=2)
        ruleset = hgp_lib.rules.rules.RuleSet([rule1, rule2], default_class=0)
        preds = ruleset.predict(data)
        np.testing.assert_array_equal(preds, np.array([1, 2, 1, 0]))
        emptyset = hgp_lib.rules.rules.RuleSet([], default_class=9)
        np.testing.assert_array_equal(emptyset.predict(data), np.array([9, 9, 9, 9]))


if __name__ == "__main__":
    unittest.main()
    # TODO: Add performance test that should execute both operator types and measure
    # Use np.testing.measure
