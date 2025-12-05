import doctest
import unittest
from unittest.mock import patch
import random

import numpy as np

import hgp_lib.crossover.crossover_executor
import hgp_lib.rules.utils
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.rules import Rule, Literal, And, Or


class TestCrossoverExecutor(unittest.TestCase):
    def test_crossover_executor_validation(self):
        with self.subTest("crossover_p type"):
            with self.assertRaises(TypeError):
                CrossoverExecutor(crossover_p=1)  # int instead of float

        with self.subTest("crossover_p bounds"):
            with self.assertRaises(ValueError):
                CrossoverExecutor(crossover_p=1.5)
            with self.assertRaises(ValueError):
                CrossoverExecutor(crossover_p=-0.1)

        with self.subTest("crossover_strategy invalid"):
            with self.assertRaises(ValueError):
                CrossoverExecutor(crossover_strategy="invalid")

        with self.subTest("num_tries requires check_valid"):
            with self.assertRaises(ValueError):
                CrossoverExecutor(num_tries=2)

        with self.subTest("check_valid must be callable returning bool"):
            with self.assertRaises(TypeError):
                CrossoverExecutor(check_valid="not callable")
            with self.assertRaises(TypeError):
                CrossoverExecutor(check_valid=lambda r: "not bool")

        with self.subTest("num_tries must be positive"):
            with self.assertRaises(ValueError):
                CrossoverExecutor(check_valid=lambda r: True, num_tries=0)

    def test_apply_random_strategy(self):
        """Test that apply selects rules and returns children."""
        executor = CrossoverExecutor(crossover_p=1.0)
        rules = [
            And([Literal(value=0), Literal(value=1)]),
            Or([Literal(value=2), Literal(value=3)]),
            And([Literal(value=4), Literal(value=5)]),
            Or([Literal(value=6), Literal(value=7)]),
        ]

        random.seed(42)
        np.random.seed(42)
        children = executor.apply(rules)

        # Should return children from crossover
        self.assertIsInstance(children, list)
        # With crossover_p=1.0, all 4 rules should be selected -> 2 pairs -> 4 children
        self.assertEqual(len(children), 4)

    def test_apply_returns_list(self):
        """Test that apply returns a list and does not modify input."""
        executor = CrossoverExecutor(crossover_p=1.0)
        rules = [
            And([Literal(value=0), Literal(value=1)]),
            Or([Literal(value=2), Literal(value=3)]),
        ]
        original_strs = [str(r) for r in rules]

        random.seed(42)
        np.random.seed(42)
        children = executor.apply(rules)

        # Original rules should be unchanged
        self.assertEqual([str(r) for r in rules], original_strs)
        # Children should be returned
        self.assertIsInstance(children, list)
        self.assertEqual(len(children), 2)

    def test_apply_empty_list(self):
        """Test that apply handles empty list."""
        executor = CrossoverExecutor(crossover_p=1.0)
        children = executor.apply([])
        self.assertEqual(children, [])

    def test_crossover_subtree_swap(self):
        """Test actual subtree exchange between two parents."""
        executor = CrossoverExecutor()

        parent_a = And([Literal(value=0), Literal(value=1)])
        parent_b = Or([Literal(value=2), Literal(value=3)])

        random.seed(42)
        children = executor.crossover(parent_a, parent_b)

        # Should return two children
        self.assertEqual(len(children), 2)
        child_a, child_b = children

        # Children should be copies, not the same objects
        self.assertIsNot(child_a, parent_a)
        self.assertIsNot(child_b, parent_b)

        # Children should still be valid Rule objects
        self.assertIsInstance(child_a, Rule)
        self.assertIsInstance(child_b, Rule)

    def test_crossover_respects_validator_and_retries(self):
        """Test that validator is called and crossover collects valid children."""
        call_count = [0]

        def validator(rule: Rule) -> bool:
            call_count[0] += 1
            # Accept every other child
            return call_count[0] % 2 == 0

        executor = CrossoverExecutor(
            check_valid=validator,
            num_tries=5,
        )

        parent_a = And([Literal(value=0), Literal(value=1)])
        parent_b = Or([Literal(value=2), Literal(value=3)])

        random.seed(0)
        children = executor.crossover(parent_a, parent_b)

        # Validator should have been called multiple times
        self.assertGreater(call_count[0], 0)
        # Should have collected up to 2 valid children
        self.assertLessEqual(len(children), 2)

    def test_crossover_returns_empty_when_validator_always_rejects(self):
        """Test that crossover returns empty list when validator always rejects."""
        def always_reject(rule: Rule) -> bool:
            return False

        executor = CrossoverExecutor(
            check_valid=always_reject,
            num_tries=3,
        )

        parent_a = And([Literal(value=0), Literal(value=1)])
        parent_b = Or([Literal(value=2), Literal(value=3)])

        random.seed(0)
        children = executor.crossover(parent_a, parent_b)

        # No children should pass validation
        self.assertEqual(len(children), 0)

    def test_crossover_collects_partial_valid_children(self):
        """Test that crossover can return 1 child if only one passes validation."""
        call_count = [0]

        def accept_first_crossover_child(rule: Rule) -> bool:
            call_count[0] += 1
            # First call is during __init__ validation, accept it
            # Then only accept first child during crossover (call 2)
            return call_count[0] <= 2

        executor = CrossoverExecutor(
            check_valid=accept_first_crossover_child,
            num_tries=1,
        )

        parent_a = And([Literal(value=0), Literal(value=1)])
        parent_b = Or([Literal(value=2), Literal(value=3)])

        random.seed(0)
        children = executor.crossover(parent_a, parent_b)

        # Only one child should pass (call 2 passes, call 3 fails)
        self.assertEqual(len(children), 1)

    def test_crossover_strategy_validation(self):
        """Test that only valid strategies are accepted."""
        # Valid strategies
        CrossoverExecutor(crossover_strategy="random")
        CrossoverExecutor(crossover_strategy="best")

        # Invalid strategy
        with self.assertRaises(ValueError):
            CrossoverExecutor(crossover_strategy="tournament")

    def test_crossover_preserves_rule_validity(self):
        """Test that crossover produces structurally valid rules."""
        executor = CrossoverExecutor()

        parent_a = And([
            Literal(value=0),
            Or([Literal(value=1), Literal(value=2)]),
        ])
        parent_b = Or([
            And([Literal(value=3), Literal(value=4)]),
            Literal(value=5),
        ])

        random.seed(123)
        children = executor.crossover(parent_a, parent_b)

        # Should get two children
        self.assertEqual(len(children), 2)

        for child in children:
            # Children should be valid Rule objects
            self.assertIsInstance(child, Rule)
            # Children should have proper structure (flatten should work)
            flat = child.flatten()
            self.assertGreater(len(flat), 0)

    def test_crossover_without_validator_always_returns_two(self):
        """Test that crossover without validator always returns exactly two children."""
        executor = CrossoverExecutor()

        parent_a = And([Literal(value=0), Literal(value=1)])
        parent_b = Or([Literal(value=2), Literal(value=3)])

        for seed in range(10):
            random.seed(seed)
            children = executor.crossover(parent_a, parent_b)
            self.assertEqual(len(children), 2, f"Failed with seed {seed}")

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.crossover.crossover_executor, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")

    def test_utils_doctests(self):
        result = doctest.testmod(hgp_lib.rules.utils, verbose=False)
        self.assertEqual(result.failed, 0, f"Utils doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
