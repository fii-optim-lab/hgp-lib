"""Tests for literal mutations: DeleteMutation, NegateMutation, ReplaceLiteral, PromoteLiteral."""

import doctest
import unittest

import numpy as np

import hgp_lib.mutations.literal_mutations
from hgp_lib.mutations import (
    DeleteMutation,
    NegateMutation,
    ReplaceLiteral,
    PromoteLiteral,
    MutationError,
)
from hgp_lib.rules import Literal, And, Or


class TestDeleteMutation(unittest.TestCase):
    def setUp(self):
        self.mutation = DeleteMutation()

    def test_flags(self):
        self.assertTrue(self.mutation.is_literal_mutation)
        self.assertTrue(self.mutation.is_operator_mutation)

    def test_simple_removal_from_3_subrules(self):
        rule = Or([Literal(value=0), Literal(value=1), Literal(value=2)])
        self.mutation.apply(rule.subrules[1])
        self.assertEqual(str(rule), "Or(0, 2)")

    def test_remove_first_subrule(self):
        rule = And([Literal(value=0), Literal(value=1), Literal(value=2)])
        self.mutation.apply(rule.subrules[0])
        self.assertEqual(str(rule), "And(1, 2)")

    def test_remove_last_subrule(self):
        rule = And([Literal(value=0), Literal(value=1), Literal(value=2)])
        self.mutation.apply(rule.subrules[2])
        self.assertEqual(str(rule), "And(0, 1)")

    def test_remove_operator_subrule(self):
        rule = Or(
            [
                Literal(value=0),
                Literal(value=1),
                And([Literal(value=2), Literal(value=3)]),
            ]
        )
        self.mutation.apply(rule.subrules[2])
        self.assertEqual(str(rule), "Or(0, 1)")

    def test_collapse_2_subrules(self):
        """Parent has 2 subrules → sibling appended to grandparent, parent removed."""
        root = Or([Literal(value=0), And([Literal(value=1), Literal(value=2)])])
        self.mutation.apply(root.subrules[1].subrules[0])
        self.assertEqual(str(root), "Or(0, 2)")

    def test_collapse_preserves_sibling_at_end(self):
        root = Or(
            [
                Literal(value=0),
                And([Literal(value=1), Literal(value=2)]),
                Literal(value=3),
            ]
        )
        self.mutation.apply(root.subrules[1].subrules[1])
        self.assertEqual(str(root), "Or(0, 3, 1)")

    def test_collapse_parent_references(self):
        root = Or([Literal(value=0), And([Literal(value=1), Literal(value=2)])])
        self.mutation.apply(root.subrules[1].subrules[0])
        # The surviving literal (value=2) should now have root as parent
        self.assertIs(root.subrules[1].parent, root)

    def test_root_node_raises(self):
        rule = Literal(value=0)
        with self.assertRaises(MutationError):
            self.mutation.apply(rule)

    def test_2_subrules_no_grandparent_raises(self):
        rule = And([Literal(value=0), Literal(value=1)])
        with self.assertRaises(MutationError):
            self.mutation.apply(rule.subrules[0])

    def test_original_unchanged_on_error(self):
        rule = Or([And([Literal(value=0), Literal(value=1)]), Literal(value=2)])
        original_str = str(rule)
        with self.assertRaises(MutationError):
            self.mutation.apply(rule)
        self.assertEqual(str(rule), original_str)


class TestNegateMutation(unittest.TestCase):
    def setUp(self):
        self.mutation = NegateMutation()

    def test_flags(self):
        self.assertTrue(self.mutation.is_literal_mutation)
        self.assertTrue(self.mutation.is_operator_mutation)

    def test_negate_literal(self):
        rule = Literal(value=0)
        self.mutation.apply(rule)
        self.assertTrue(rule.negated)
        self.assertEqual(str(rule), "~0")

    def test_double_negate_literal(self):
        rule = Literal(value=0)
        self.mutation.apply(rule)
        self.mutation.apply(rule)
        self.assertFalse(rule.negated)

    def test_negate_operator(self):
        rule = And([Literal(value=0), Literal(value=1)])
        self.mutation.apply(rule)
        self.assertTrue(rule.negated)
        self.assertEqual(str(rule), "~And(0, 1)")

    def test_double_negate_operator(self):
        rule = Or([Literal(value=0), Literal(value=1)])
        self.mutation.apply(rule)
        self.mutation.apply(rule)
        self.assertFalse(rule.negated)

    def test_negate_already_negated(self):
        rule = Literal(value=0, negated=True)
        self.mutation.apply(rule)
        self.assertFalse(rule.negated)


class TestReplaceLiteral(unittest.TestCase):
    def test_flags(self):
        m = ReplaceLiteral(num_literals=5)
        self.assertTrue(m.is_literal_mutation)
        self.assertFalse(m.is_operator_mutation)

    def test_value_changes(self):
        m = ReplaceLiteral(num_literals=5)
        rule = Literal(value=0)
        m.apply(rule)
        self.assertNotEqual(rule.value, 0)

    def test_value_in_range(self):
        m = ReplaceLiteral(num_literals=5)
        rule = Literal(value=2)
        m.apply(rule)
        self.assertGreaterEqual(rule.value, 0)
        self.assertLess(rule.value, 5)

    def test_two_literals_toggles(self):
        m = ReplaceLiteral(num_literals=2)
        rule = Literal(value=0)
        m.apply(rule)
        self.assertEqual(rule.value, 1)
        m.apply(rule)
        self.assertEqual(rule.value, 0)

    def test_negation_preserved(self):
        m = ReplaceLiteral(num_literals=5)
        rule = Literal(value=0, negated=True)
        m.apply(rule)
        self.assertTrue(rule.negated)


class TestPromoteLiteral(unittest.TestCase):
    def test_flags(self):
        m = PromoteLiteral(num_literals=4)
        self.assertTrue(m.is_literal_mutation)
        self.assertFalse(m.is_operator_mutation)

    def test_becomes_operator(self):
        np.random.seed(42)
        m = PromoteLiteral(num_literals=4, operator_types=(Or, And))
        rule = Literal(value=1)
        m.apply(rule)
        self.assertIn(type(rule), [Or, And])
        self.assertIsNone(rule.value)

    def test_has_two_subrules(self):
        np.random.seed(42)
        m = PromoteLiteral(num_literals=4)
        rule = Literal(value=1)
        m.apply(rule)
        self.assertEqual(len(rule.subrules), 2)

    def test_original_literal_preserved(self):
        np.random.seed(42)
        m = PromoteLiteral(num_literals=4)
        rule = Literal(value=2, negated=True)
        m.apply(rule)
        self.assertEqual(rule.subrules[0].value, 2)
        self.assertTrue(rule.subrules[0].negated)

    def test_new_literal_different_value(self):
        np.random.seed(42)
        m = PromoteLiteral(num_literals=4)
        rule = Literal(value=2)
        m.apply(rule)
        self.assertNotEqual(rule.subrules[1].value, 2)

    def test_complexity_increases(self):
        np.random.seed(42)
        m = PromoteLiteral(num_literals=4)
        rule = Literal(value=0)
        self.assertEqual(len(rule), 1)
        m.apply(rule)
        self.assertEqual(len(rule), 3)

    def test_parent_references(self):
        np.random.seed(42)
        m = PromoteLiteral(num_literals=4)
        rule = Literal(value=0)
        m.apply(rule)
        for s in rule.subrules:
            self.assertIs(s.parent, rule)


class TestDoctests(unittest.TestCase):
    def test_doctests(self):
        result = doctest.testmod(hgp_lib.mutations.literal_mutations, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
