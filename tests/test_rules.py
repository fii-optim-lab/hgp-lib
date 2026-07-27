import os
import subprocess
import sys
import unittest

import numpy as np

from hgp_lib.rules import Literal
from hgp_lib.rules.low_memory_operators import And as LowMemoryAnd
from hgp_lib.rules.low_memory_operators import Or as LowMemoryOr
from hgp_lib.rules.operators import And, Or


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

    def test_to_str_indented_multiline(self):
        rule = And([Literal(value=0), Literal(value=1)])
        single_line = rule.to_str()
        multiline = rule.to_str(indent=0)
        # Indented output spans multiple lines and uses tab indentation.
        self.assertIn("\n", multiline)
        self.assertIn("\t", multiline)
        self.assertNotIn("\n", single_line)
        # Named features are still substituted in the indented form.
        named = rule.to_str(["a", "b"], indent=0)
        self.assertIn("a", named)
        self.assertIn("b", named)

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
                    np.zeros(len(self.data), dtype=bool),
                )

                all_one = and_type(
                    subrules=[
                        Literal(value=3),
                        Literal(value=3, negated=True),
                    ],
                    negated=True,
                )
                np.testing.assert_array_equal(
                    all_one.evaluate(self.data), np.ones(len(self.data), dtype=bool)
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
                        Literal(value=0, negated=True),
                        Literal(value=1, negated=True),
                        Literal(value=4),
                    ],
                    negated=True,
                )
                result = ~(~self.data[:, 0] & ~self.data[:, 1] & self.data[:, 4])
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
                    all_one.evaluate(self.data), np.ones(len(self.data), dtype=bool)
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
                    np.zeros(len(self.data), dtype=bool),
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
                        Literal(value=0, negated=True),
                        Literal(value=1, negated=True),
                        Literal(value=4),
                    ],
                    negated=True,
                )
                result = ~(~self.data[:, 0] | ~self.data[:, 1] | self.data[:, 4])
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


class TestImports(unittest.TestCase):
    def run_import_test(self, low_memory, expected_module):
        env = os.environ.copy()
        env["HGP_LOW_MEMORY"] = str(low_memory)

        code = (
            "import hgp_lib\n"
            f'assert hgp_lib.rules.Or.__module__ == "{expected_module}"\n'
            f'assert hgp_lib.rules.And.__module__ == "{expected_module}"'
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=result.stderr,
        )

    def test_normal_operators_imported(self):
        self.run_import_test(0, "hgp_lib.rules.operators")

    def test_low_memory_operators_imported(self):
        self.run_import_test(1, "hgp_lib.rules.low_memory_operators")


if __name__ == "__main__":
    unittest.main()
    # TODO: Add performance test that should execute both operator types and measure
    # Use np.testing.measure
