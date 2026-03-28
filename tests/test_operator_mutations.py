"""Tests for operator mutations: RemoveIntermediateOperator, ReplaceOperator, AddLiteral."""

import doctest
import unittest


import hgp_lib.mutations.operator_mutations
from hgp_lib.mutations import (
    RemoveIntermediateOperator,
    ReplaceOperator,
    AddLiteral,
    MutationError,
)
from hgp_lib.rules import Literal, And, Or


class TestRemoveIntermediateOperator(unittest.TestCase):
    def setUp(self):
        self.mutation = RemoveIntermediateOperator()

    def test_flags(self):
        self.assertFalse(self.mutation.is_literal_mutation)
        self.assertTrue(self.mutation.is_operator_mutation)

    def test_removes_operator_and_promotes_children(self):
        rule = And(
            [
                Literal(value=0),
                Or([Literal(value=1), Literal(value=2)]),
                Literal(value=3),
            ]
        )
        self.mutation.apply(rule.subrules[1])
        self.assertEqual(str(rule), "And(0, 3, 1, 2)")

    def test_children_reparented(self):
        rule = And([Literal(value=0), Or([Literal(value=1), Literal(value=2)])])
        inner = rule.subrules[1]
        children = list(inner.subrules)
        self.mutation.apply(inner)
        for child in children:
            self.assertIs(child.parent, rule)

    def test_root_raises(self):
        rule = And([Literal(value=0), Literal(value=1)])
        with self.assertRaises(MutationError):
            self.mutation.apply(rule)

    def test_root_unchanged_on_error(self):
        rule = And([Literal(value=0), Literal(value=1)])
        original = str(rule)
        with self.assertRaises(MutationError):
            self.mutation.apply(rule)
        self.assertEqual(str(rule), original)

    def test_nested_removal(self):
        rule = And(
            [
                Or([Literal(value=0), Literal(value=1)]),
                And([Literal(value=2), Literal(value=3)]),
                Literal(value=4),
            ]
        )
        self.mutation.apply(rule.subrules[0])
        self.assertEqual(str(rule), "And(And(2, 3), 4, 0, 1)")
        self.mutation.apply(rule.subrules[0])
        self.assertEqual(str(rule), "And(4, 0, 1, 2, 3)")


class TestReplaceOperator(unittest.TestCase):
    def test_flags(self):
        m = ReplaceOperator()
        self.assertFalse(m.is_literal_mutation)
        self.assertTrue(m.is_operator_mutation)

    def test_and_to_or(self):
        rule = And([Literal(value=0), Literal(value=1)])
        ReplaceOperator().apply(rule)
        self.assertEqual(type(rule), Or)

    def test_or_to_and(self):
        rule = Or([Literal(value=0), Literal(value=1)])
        ReplaceOperator().apply(rule)
        self.assertEqual(type(rule), And)

    def test_preserves_negation(self):
        rule = Or([Literal(value=0), Literal(value=1)], negated=True)
        ReplaceOperator().apply(rule)
        self.assertTrue(rule.negated)

    def test_preserves_subrules(self):
        rule = And([Literal(value=0), Literal(value=1), Literal(value=2)])
        ReplaceOperator().apply(rule)
        self.assertEqual(len(rule.subrules), 3)

    def test_double_replace_restores(self):
        rule = And([Literal(value=0), Literal(value=1)])
        m = ReplaceOperator()
        m.apply(rule)
        self.assertEqual(type(rule), Or)
        m.apply(rule)
        self.assertEqual(type(rule), And)


class TestAddLiteral(unittest.TestCase):
    def test_flags(self):
        m = AddLiteral(num_literals=5)
        self.assertFalse(m.is_literal_mutation)
        self.assertTrue(m.is_operator_mutation)

    def test_adds_one_subrule(self):
        rule = And([Literal(value=0), Literal(value=1)])
        AddLiteral(num_literals=5).apply(rule)
        self.assertEqual(len(rule.subrules), 3)

    def test_new_literal_in_range(self):
        rule = And([Literal(value=0)])
        AddLiteral(num_literals=5).apply(rule)
        new_val = rule.subrules[1].value
        self.assertGreaterEqual(new_val, 0)
        self.assertLess(new_val, 5)

    def test_new_literal_not_duplicate(self):
        rule = Or([Literal(value=0), Literal(value=1)])
        AddLiteral(num_literals=3).apply(rule)
        self.assertEqual(rule.subrules[2].value, 2)

    def test_all_literals_present_raises(self):
        rule = Or([Literal(value=0), Literal(value=1), Literal(value=2)])
        with self.assertRaises(MutationError):
            AddLiteral(num_literals=3).apply(rule)

    def test_unchanged_on_error(self):
        rule = And([Literal(value=0), Literal(value=1)])
        original = str(rule)
        with self.assertRaises(MutationError):
            AddLiteral(num_literals=2).apply(rule)
        self.assertEqual(str(rule), original)

    def test_fills_all_slots(self):
        m = AddLiteral(num_literals=5)
        rule = Or([Literal(value=0)])
        for _ in range(4):
            m.apply(rule)
        values = {s.value for s in rule.subrules}
        self.assertEqual(values, {0, 1, 2, 3, 4})

    def test_parent_reference_set(self):
        rule = And([Literal(value=0)])
        AddLiteral(num_literals=5).apply(rule)
        self.assertIs(rule.subrules[1].parent, rule)


class TestDoctests(unittest.TestCase):
    def test_doctests(self):
        result = doctest.testmod(hgp_lib.mutations.operator_mutations, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
