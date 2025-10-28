import doctest
import unittest
import numpy as np

import hgp_lib
from hgp_lib.rules import Literal
from hgp_lib.rules.operators import And, Or
from hgp_lib.rules.low_memory_operators import And as LowMemoryAnd, Or as LowMemoryOr


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

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.rules.rules, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.rules.literals, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.rules.operators, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.rules.low_memory_operators, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
    # TODO: Add performance test that should execute both operator types and measure
    # Use np.testing.measure
