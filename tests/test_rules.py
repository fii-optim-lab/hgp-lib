import unittest

import numpy as np

import hgp_lib
from hgp_lib.evaluation import NumpyBackend
from hgp_lib.rules import Literal
from hgp_lib.rules.operators import And, Or

# Both NumPy evaluation algorithms must give the same results.
BACKENDS = (NumpyBackend(), NumpyBackend(low_memory=True))


class TestRules(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(10, 20) < 0.5

    def assert_evaluates_to(self, rule, expected):
        for backend in BACKENDS:
            with self.subTest(low_memory=backend.low_memory):
                np.testing.assert_array_equal(
                    backend.predict(rule, self.data), expected
                )

    def test_literal(self):
        self.assert_evaluates_to(Literal(value=0, negated=True), ~self.data[:, 0])
        self.assert_evaluates_to(Literal(value=0, negated=False), self.data[:, 0])
        self.assert_evaluates_to(Literal(value=1, negated=False), self.data[:, 1])
        self.assert_evaluates_to(Literal(value=1, negated=True), ~self.data[:, 1])

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
        data = self.data
        self.assert_evaluates_to(And([Literal(value=0), Literal(value=0)]), data[:, 0])
        self.assert_evaluates_to(
            And([Literal(value=1), Literal(value=1)], negated=True), ~data[:, 1]
        )
        self.assert_evaluates_to(
            And([Literal(value=2), Literal(value=2, negated=True)]),
            np.zeros(len(data), dtype=bool),
        )
        self.assert_evaluates_to(
            And([Literal(value=3), Literal(value=3, negated=True)], negated=True),
            np.ones(len(data), dtype=bool),
        )
        self.assert_evaluates_to(
            And([Literal(value=0), Literal(value=1, negated=True), Literal(value=2)]),
            data[:, 0] & ~data[:, 1] & data[:, 2],
        )
        self.assert_evaluates_to(
            And(
                [
                    Literal(value=0, negated=True),
                    Literal(value=1, negated=True),
                    Literal(value=4),
                ],
                negated=True,
            ),
            ~(~data[:, 0] & ~data[:, 1] & data[:, 4]),
        )

    def test_or(self):
        data = self.data
        self.assert_evaluates_to(Or([Literal(value=0), Literal(value=0)]), data[:, 0])
        self.assert_evaluates_to(
            Or([Literal(value=1), Literal(value=1)], negated=True), ~data[:, 1]
        )
        self.assert_evaluates_to(
            Or([Literal(value=2), Literal(value=2, negated=True)]),
            np.ones(len(data), dtype=bool),
        )
        self.assert_evaluates_to(
            Or([Literal(value=3), Literal(value=3, negated=True)], negated=True),
            np.zeros(len(data), dtype=bool),
        )
        self.assert_evaluates_to(
            Or([Literal(value=0), Literal(value=1, negated=True), Literal(value=2)]),
            data[:, 0] | ~data[:, 1] | data[:, 2],
        )
        self.assert_evaluates_to(
            Or(
                [
                    Literal(value=0, negated=True),
                    Literal(value=1, negated=True),
                    Literal(value=4),
                ],
                negated=True,
            ),
            ~(~data[:, 0] | ~data[:, 1] | data[:, 4]),
        )

    def test_operators(self):
        rule = Or(
            subrules=[
                And(subrules=[Literal(value=1), Literal(value=2, negated=True)]),
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
        data = self.data
        self.assert_evaluates_to(
            rule,
            data[:, 1] & ~data[:, 2]
            | (data[:, 3] | data[:, 4] | ~data[:, 5]) & ~data[:, 6]
            | data[:, 7]
            | ~data[:, 8],
        )


class TestImports(unittest.TestCase):
    def test_operators_come_from_one_module(self):
        self.assertEqual(hgp_lib.rules.And.__module__, "hgp_lib.rules.operators")
        self.assertEqual(hgp_lib.rules.Or.__module__, "hgp_lib.rules.operators")
        self.assertFalse(hasattr(hgp_lib.rules, "low_memory_operators"))


if __name__ == "__main__":
    unittest.main()
