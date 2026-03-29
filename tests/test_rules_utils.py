"""Tests for hgp_lib.rules.utils."""

import doctest
import random
import unittest

import hgp_lib.rules.utils
from hgp_lib.rules import Literal, And, Or
from hgp_lib.rules.utils import (
    is_operator,
    is_operator_type,
    replace_with_rule,
    deep_swap,
    apply_feature_mapping,
    select_crossover_point,
)


class TestIsOperator(unittest.TestCase):
    def test_and_is_operator(self):
        self.assertTrue(is_operator(And([Literal(value=0), Literal(value=1)])))

    def test_or_is_operator(self):
        self.assertTrue(is_operator(Or([Literal(value=0), Literal(value=1)])))

    def test_literal_is_not_operator(self):
        self.assertFalse(is_operator(Literal(value=0)))

    def test_non_rule_is_not_operator(self):
        self.assertFalse(is_operator("not a rule"))

    def test_negated_operator(self):
        self.assertTrue(is_operator(And([Literal(value=0)], negated=True)))


class TestIsOperatorType(unittest.TestCase):
    def test_and_type(self):
        self.assertTrue(is_operator_type(And))

    def test_or_type(self):
        self.assertTrue(is_operator_type(Or))

    def test_literal_type(self):
        self.assertFalse(is_operator_type(Literal))

    def test_non_rule_type(self):
        self.assertFalse(is_operator_type(str))

    def test_non_type(self):
        self.assertFalse(is_operator_type(42))


class TestReplaceWithRule(unittest.TestCase):
    def test_and_to_or(self):
        target = And([Literal(value=0), Literal(value=1)])
        source = Or([Literal(value=2), Literal(value=3)])
        original_id = id(target)
        replace_with_rule(target, source)
        self.assertEqual(id(target), original_id)
        self.assertEqual(type(target), Or)
        self.assertEqual(str(target), "Or(2, 3)")
        for subrule in target.subrules:
            self.assertEqual(subrule.parent, target)

    def test_or_to_and_with_negation(self):
        target = Or([Literal(value=0)], negated=False)
        source = And([Literal(value=1), Literal(value=2)], negated=True)
        replace_with_rule(target, source)
        self.assertEqual(type(target), And)
        self.assertTrue(target.negated)
        self.assertEqual(str(target), "~And(1, 2)")

    def test_operator_to_literal(self):
        target = And([Literal(value=0), Literal(value=1)])
        source = Literal(value=5, negated=True)
        replace_with_rule(target, source)
        self.assertEqual(type(target), Literal)
        self.assertEqual(target.value, 5)
        self.assertTrue(target.negated)

    def test_literal_to_operator(self):
        target = Literal(value=0)
        source = Or([Literal(value=1), Literal(value=2)])
        replace_with_rule(target, source)
        self.assertEqual(type(target), Or)
        self.assertIsNone(target.value)
        self.assertEqual(len(target.subrules), 2)

    def test_source_unchanged(self):
        target = And([Literal(value=0)])
        source = Or([Literal(value=1), Literal(value=2)])
        source_str = str(source)
        replace_with_rule(target, source)
        self.assertEqual(str(source), source_str)


class TestDeepSwap(unittest.TestCase):
    def test_swap_and_or(self):
        a = And([Literal(value=0), Literal(value=1)])
        b = Or([Literal(value=2), Literal(value=3)])
        id_a, id_b = id(a), id(b)
        deep_swap(a, b)
        self.assertEqual(id(a), id_a)
        self.assertEqual(id(b), id_b)
        self.assertEqual(type(a), Or)
        self.assertEqual(type(b), And)
        self.assertEqual(str(a), "Or(2, 3)")
        self.assertEqual(str(b), "And(0, 1)")

    def test_swap_negation(self):
        a = And([Literal(value=0)], negated=True)
        b = Or([Literal(value=1)], negated=False)
        deep_swap(a, b)
        self.assertFalse(a.negated)
        self.assertTrue(b.negated)

    def test_swap_literal_and_operator(self):
        a = Literal(value=5, negated=True)
        b = And([Literal(value=0), Literal(value=1)])
        deep_swap(a, b)
        self.assertEqual(type(a), And)
        self.assertEqual(str(a), "And(0, 1)")
        self.assertEqual(type(b), Literal)
        self.assertEqual(b.value, 5)
        self.assertTrue(b.negated)

    def test_swap_two_literals(self):
        a = Literal(value=0, negated=False)
        b = Literal(value=1, negated=True)
        deep_swap(a, b)
        self.assertEqual(a.value, 1)
        self.assertTrue(a.negated)
        self.assertEqual(b.value, 0)
        self.assertFalse(b.negated)

    def test_parent_references_after_swap(self):
        a = And([Literal(value=0), Literal(value=1)])
        b = Or([Literal(value=2), Literal(value=3)])
        deep_swap(a, b)
        for s in a.subrules:
            self.assertEqual(s.parent, a)
        for s in b.subrules:
            self.assertEqual(s.parent, b)


class TestApplyFeatureMapping(unittest.TestCase):
    def test_none_mapping_returns_same_object(self):
        rule = And([Literal(value=0), Literal(value=1)])
        result = apply_feature_mapping(rule, None)
        self.assertIs(result, rule)

    def test_mapping_returns_copy(self):
        rule = And([Literal(value=0), Literal(value=1)])
        result = apply_feature_mapping(rule, {0: 5, 1: 10})
        self.assertIsNot(result, rule)

    def test_mapping_applied(self):
        rule = And([Literal(value=0), Literal(value=1)])
        result = apply_feature_mapping(rule, {0: 5, 1: 10})
        self.assertEqual(str(result), "And(5, 10)")

    def test_original_unchanged(self):
        rule = And([Literal(value=0), Literal(value=1)])
        apply_feature_mapping(rule, {0: 5, 1: 10})
        self.assertEqual(str(rule), "And(0, 1)")

    def test_nested_mapping(self):
        rule = Or([And([Literal(value=0), Literal(value=1)]), Literal(value=2)])
        result = apply_feature_mapping(rule, {0: 10, 1: 20, 2: 30})
        self.assertEqual(str(result), "Or(And(10, 20), 30)")

    def test_single_literal(self):
        rule = Literal(value=3)
        result = apply_feature_mapping(rule, {3: 99})
        self.assertEqual(result.value, 99)
        self.assertEqual(rule.value, 3)


class TestSelectCrossoverPoint(unittest.TestCase):
    def test_single_literal(self):
        rule = Literal(value=0)
        selected = select_crossover_point(rule)
        self.assertIsInstance(selected, Literal)

    def test_operator_p_1_selects_operator(self):
        random.seed(42)
        rule = And([Literal(value=0), Or([Literal(value=1), Literal(value=2)])])
        selected = select_crossover_point(rule, operator_p=1.0)
        self.assertTrue(is_operator(selected))

    def test_operator_p_0_selects_literal(self):
        random.seed(42)
        rule = And([Literal(value=0), Or([Literal(value=1), Literal(value=2)])])
        selected = select_crossover_point(rule, operator_p=0.0)
        self.assertIsInstance(selected, Literal)

    def test_returns_node_from_tree(self):
        rule = And([Literal(value=0), Or([Literal(value=1), Literal(value=2)])])
        all_nodes = rule.flatten()
        for _ in range(20):
            selected = select_crossover_point(rule)
            self.assertIn(selected, all_nodes)

    def test_all_literals_tree(self):
        """When tree has no operators except root, operator_p=1.0 still returns root."""
        rule = And([Literal(value=0), Literal(value=1)])
        random.seed(0)
        selected = select_crossover_point(rule, operator_p=1.0)
        self.assertIs(selected, rule)

    def test_distribution_biased(self):
        """With operator_p=0.9, operators should be selected much more often than literals."""
        random.seed(42)
        rule = And(
            [
                Literal(value=0),
                Or([Literal(value=1), Literal(value=2)]),
                Literal(value=3),
            ]
        )
        operator_count = 0
        n = 200
        for _ in range(n):
            selected = select_crossover_point(rule, operator_p=0.9)
            if is_operator(selected):
                operator_count += 1
        # With operator_p=0.9, expect ~90% operators
        self.assertGreater(operator_count / n, 0.7)


class TestDoctests(unittest.TestCase):
    def test_doctests(self):
        result = doctest.testmod(hgp_lib.rules.utils, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
