import doctest
import unittest

import hgp_lib.mutations
from hgp_lib.mutations import (
    Mutation,
    DeleteMutation,
    MutationError,
    NegateMutation,
    ReplaceLiteral,
    PromoteLiteral,
    RemoveIntermediateOperator,
    ReplaceOperator,
    AddLiteral,
)
from hgp_lib.rules import Rule, Literal, Or, And


class TestMutations(unittest.TestCase):
    def test_base_mutation(self):
        with self.subTest("Bad Mutation Init"):

            class BadMutation(Mutation):
                def __init__(self):
                    super().__init__(False, False)

                def apply(self, rule: Rule):
                    pass

            with self.assertRaises(ValueError) as e:
                BadMutation()
            self.assertIn(
                "A mutation must be at least either a literal mutation, or an operator mutation.",
                str(e.exception),
            )

        with self.subTest("Good Mutation Init"):

            class GoodMutation(Mutation):
                def __init__(self):
                    super().__init__(True, False)

                def apply(self, rule: Rule):
                    pass

            GoodMutation().apply(Literal(value=0))

    def test_delete_mutation(self):
        rule = Or(
            [
                And(
                    [
                        Literal(value=0),
                        Literal(value=1),
                    ]
                ),
                Literal(value=1),
                Literal(value=2),
            ]
        )
        mutation = DeleteMutation()

        self.assertTrue(mutation.is_literal_mutation)
        self.assertTrue(mutation.is_operator_mutation)

        with self.subTest("Normal behavior"):
            copy = rule.copy()
            mutation.apply(copy.subrules[0])
            self.assertEqual(str(copy), "Or(1, 2)")

            copy = rule.copy()
            mutation.apply(copy.subrules[1])
            self.assertEqual(str(copy), "Or(And(0, 1), 2)")

        with self.subTest("Mutation fails if invalid"):
            with self.assertRaises(MutationError):
                mutation.apply(rule)
            self.assertEqual(str(rule), "Or(And(0, 1), 1, 2)")

            with self.assertRaises(MutationError):
                mutation.apply(rule.subrules[0].subrules[0])
            self.assertEqual(str(rule), "Or(And(0, 1), 1, 2)")

            mutation.apply(rule.subrules[1])
            with self.assertRaises(MutationError):
                mutation.apply(rule.subrules[1])
            self.assertEqual(str(rule), "Or(And(0, 1), 2)")

    def test_negate_mutation(self):
        rule = Or(
            [
                And(
                    [
                        Literal(value=0),
                        Literal(value=1),
                    ]
                ),
                Literal(value=1),
                Literal(value=2),
            ]
        )
        mutation = NegateMutation()

        self.assertTrue(mutation.is_literal_mutation)
        self.assertTrue(mutation.is_operator_mutation)

        mutation.apply(rule.subrules[2])
        self.assertEqual(str(rule), "Or(And(0, 1), 1, ~2)")
        self.assertTrue(rule.subrules[2].negated)

        mutation.apply(rule.subrules[2])
        self.assertEqual(str(rule), "Or(And(0, 1), 1, 2)")
        self.assertFalse(rule.subrules[2].negated)

        mutation.apply(rule)
        self.assertEqual(str(rule), "~Or(And(0, 1), 1, 2)")
        self.assertTrue(rule.negated)

        mutation.apply(rule)
        self.assertEqual(str(rule), "Or(And(0, 1), 1, 2)")
        self.assertFalse(rule.negated)

    def test_replace_literal(self):
        with self.subTest("Validate Constructor"):
            with self.assertRaises(TypeError):
                mutation = ReplaceLiteral()
            with self.assertRaises(TypeError):
                mutation = ReplaceLiteral("x")
            with self.assertRaises(TypeError):
                mutation = ReplaceLiteral(1.0)
            with self.assertRaises(TypeError):
                mutation = ReplaceLiteral([1])

            with self.assertRaises(ValueError):
                mutation = ReplaceLiteral(-1)

        rule = Or(
            [
                And(
                    [
                        Literal(value=0),
                        Literal(value=1),
                    ]
                ),
                Literal(value=1),
            ]
        )
        mutation = ReplaceLiteral(2)

        self.assertTrue(mutation.is_literal_mutation)
        self.assertFalse(mutation.is_operator_mutation)

        mutation.apply(rule.subrules[1])
        self.assertEqual(rule.subrules[1].value, 0)

        mutation.apply(rule.subrules[1])
        self.assertEqual(rule.subrules[1].value, 1)

        mutation = ReplaceLiteral(5)
        old_value = rule.subrules[0].subrules[0].value
        mutation.apply(rule.subrules[0].subrules[0])
        self.assertNotEqual(rule.subrules[0].subrules[0].value, old_value)
        self.assertGreaterEqual(rule.subrules[0].subrules[0].value, 0)
        self.assertLess(rule.subrules[0].subrules[0].value, 5)

    def test_promote_literal(self):
        with self.subTest("Validate Constructor"):
            with self.assertRaises(TypeError):
                mutation = PromoteLiteral()
            with self.assertRaises(TypeError):
                mutation = PromoteLiteral("x")
            with self.assertRaises(TypeError):
                mutation = PromoteLiteral(1.0)
            with self.assertRaises(TypeError):
                mutation = PromoteLiteral([1])

            with self.assertRaises(ValueError):
                mutation = PromoteLiteral(-1)

            with self.assertRaises(TypeError):
                mutation = PromoteLiteral(1, dict())

            with self.assertRaises(ValueError):
                mutation = PromoteLiteral(1, [])
            with self.assertRaises(ValueError):
                mutation = PromoteLiteral(1, [Or])

            with self.assertRaises(TypeError):
                mutation = PromoteLiteral(1, [Or, 1])

        mutation = PromoteLiteral(2, [Or, And])

        self.assertTrue(mutation.is_literal_mutation)
        self.assertFalse(mutation.is_operator_mutation)

        rule = Literal(value=0)
        mutation.apply(rule)
        self.assertEqual(len(rule), 3)
        self.assertIn(type(rule), [Or, And])
        self.assertIsNone(rule.value)
        self.assertEqual(str(Literal(value=0)), str(rule.subrules[0]))
        self.assertEqual(rule.subrules[1].value, 1)

    def test_remove_intermediate_operator(self):
        mutation = RemoveIntermediateOperator()
        rule = And(
            [
                Or([Literal(value=0), Literal(value=1)]),
                And(
                    [
                        Literal(value=2),
                        Literal(value=3),
                    ]
                ),
                Literal(value=4),
            ]
        )
        rule_str = str(rule)

        self.assertFalse(mutation.is_literal_mutation)
        self.assertTrue(mutation.is_operator_mutation)

        with self.subTest("Mutation fails if invalid"):
            with self.assertRaises(MutationError):
                mutation.apply(rule)
            self.assertEqual(str(rule), rule_str)

        mutation.apply(rule.subrules[0])
        self.assertEqual(str(rule), "And(And(2, 3), 4, 0, 1)")

        mutation.apply(rule.subrules[0])
        self.assertEqual(str(rule), "And(4, 0, 1, 2, 3)")

    def test_replace_operator(self):
        with self.subTest("Validate Constructor"):
            with self.assertRaises(TypeError):
                mutation = ReplaceOperator(1)
            with self.assertRaises(TypeError):
                mutation = ReplaceOperator(dict())

            with self.assertRaises(ValueError):
                mutation = ReplaceOperator([])
            with self.assertRaises(ValueError):
                mutation = ReplaceOperator([Or])

            with self.assertRaises(TypeError):
                mutation = ReplaceOperator([Or, 1])

        mutation = ReplaceOperator()

        self.assertFalse(mutation.is_literal_mutation)
        self.assertTrue(mutation.is_operator_mutation)

        rule = Or(
            [
                Literal(value=0),
                Literal(value=1),
            ],
            negated=True,
        )

        mutation.apply(rule)
        self.assertEqual(type(rule), And)
        self.assertTrue(rule.negated)

        mutation.apply(rule)
        self.assertEqual(type(rule), Or)
        self.assertTrue(rule.negated)

    def test_add_literal(self):
        with self.subTest("Validate Constructor"):
            with self.assertRaises(TypeError):
                mutation = AddLiteral()
            with self.assertRaises(TypeError):
                mutation = AddLiteral("x")
            with self.assertRaises(TypeError):
                mutation = AddLiteral(1.0)
            with self.assertRaises(TypeError):
                mutation = AddLiteral([1])

            with self.assertRaises(ValueError):
                mutation = AddLiteral(-1)

        mutation = AddLiteral(5)

        self.assertFalse(mutation.is_literal_mutation)
        self.assertTrue(mutation.is_operator_mutation)

        rule = Or(
            [
                And(
                    [
                        Literal(value=0),
                        Literal(value=1),
                    ]
                ),
                Literal(value=0),
                Literal(value=1),
                Literal(value=2),
                Literal(value=3),
                Literal(value=4),
            ]
        )

        copy = rule.copy()
        with self.assertRaises(MutationError):
            mutation.apply(rule)
        self.assertEqual(str(rule), str(copy))

        mutation.apply(rule.subrules[0])
        mutation.apply(rule.subrules[0])
        mutation.apply(rule.subrules[0])

        self.assertEqual({0, 1, 2, 3, 4}, {x.value for x in rule.subrules[0].subrules})

        copy = rule.copy()
        with self.assertRaises(MutationError):
            mutation.apply(rule.subrules[0])
        self.assertEqual(str(rule), str(copy))

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.mutations.base_mutation, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.mutations.literal_mutations, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.mutations.operator_mutations, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.mutations.mutation_executor, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
