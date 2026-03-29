"""Tests for MutationExecutor functionality."""

import doctest
import unittest
from unittest.mock import patch


import hgp_lib.mutations.mutation_executor
from hgp_lib.mutations import Mutation, MutationExecutor, NegateMutation, MutationError
from hgp_lib.rules import Rule, Literal, And


def _fake_select_first(rule, operator_p=0.9):
    """Always select the root node."""
    return rule.flatten()[0]


class _IncrementLiteral(Mutation):
    def __init__(self):
        super().__init__(True, False)

    def apply(self, rule: Rule):
        rule.value += 1


class _ToggleOperator(Mutation):
    def __init__(self):
        super().__init__(False, True)

    def apply(self, rule: Rule):
        rule.negated = not rule.negated


class _FailingMutation(Mutation):
    def __init__(self):
        super().__init__(True, False)

    def apply(self, rule: Rule):
        raise MutationError("always fails")


class TestMutationExecutor(unittest.TestCase):
    def _patch(self):
        """Context manager that patches random.choice and select_crossover_point."""
        return (
            patch(
                "hgp_lib.mutations.mutation_executor.random.choice",
                side_effect=lambda seq: seq[0],
            ),
            patch(
                "hgp_lib.mutations.mutation_executor.select_crossover_point",
                side_effect=_fake_select_first,
            ),
        )

    # ------------------------------------------------------------------ #
    #  apply — basic
    # ------------------------------------------------------------------ #
    def test_apply_literal(self):
        executor = MutationExecutor(
            literal_mutations=[_IncrementLiteral()],
            operator_mutations=[_ToggleOperator()],
            mutation_p=1.0,
        )
        rules = [Literal(value=0)]
        p1, p2 = self._patch()
        with p1, p2:
            executor.apply(rules)
        self.assertEqual(rules[0].value, 1)

    def test_apply_operator(self):
        executor = MutationExecutor(
            literal_mutations=[_IncrementLiteral()],
            operator_mutations=[_ToggleOperator()],
            mutation_p=1.0,
        )
        rules = [And([Literal(value=0), Literal(value=1)])]
        p1, p2 = self._patch()
        with p1, p2:
            executor.apply(rules)
        self.assertTrue(rules[0].negated)

    def test_apply_preserves_list_length(self):
        executor = MutationExecutor(
            literal_mutations=[NegateMutation()],
            operator_mutations=[NegateMutation()],
            mutation_p=1.0,
        )
        rules = [Literal(value=i) for i in range(5)]
        p1, p2 = self._patch()
        with p1, p2:
            executor.apply(rules)
        self.assertEqual(len(rules), 5)

    def test_apply_zero_mutation_p(self):
        executor = MutationExecutor(
            literal_mutations=[_IncrementLiteral()],
            operator_mutations=[_ToggleOperator()],
            mutation_p=0.0,
        )
        rules = [Literal(value=0)]
        executor.apply(rules)
        self.assertEqual(rules[0].value, 0)

    # ------------------------------------------------------------------ #
    #  apply — MutationError handling
    # ------------------------------------------------------------------ #
    def test_mutation_error_keeps_original(self):
        executor = MutationExecutor(
            literal_mutations=[_FailingMutation()],
            operator_mutations=[_ToggleOperator()],
            mutation_p=1.0,
        )
        rules = [Literal(value=5)]
        p1, p2 = self._patch()
        with p1, p2:
            executor.apply(rules)
        self.assertEqual(rules[0].value, 5)

    # ------------------------------------------------------------------ #
    #  apply — check_valid and retries
    # ------------------------------------------------------------------ #
    def test_validator_rejects_mutation(self):
        """When validator rejects, original rule is kept."""

        def even_only(rule: Rule):
            return rule.value is None or rule.value % 2 == 0

        executor = MutationExecutor(
            literal_mutations=[_IncrementLiteral()],
            operator_mutations=[_ToggleOperator()],
            mutation_p=1.0,
            check_valid=even_only,
            num_tries=2,
        )
        rules = [Literal(value=2)]
        p1, p2 = self._patch()
        with p1, p2:
            executor.apply(rules)
        # Increment 2→3 (odd, rejected), retry 3→4 (even, but that's on a fresh copy
        # so it goes 2→3 again). Both tries fail → original kept.
        self.assertEqual(rules[0].value, 2)

    def test_validator_accepts_mutation(self):
        def always_valid(rule: Rule):
            return True

        executor = MutationExecutor(
            literal_mutations=[_IncrementLiteral()],
            operator_mutations=[_ToggleOperator()],
            mutation_p=1.0,
            check_valid=always_valid,
            num_tries=1,
        )
        rules = [Literal(value=0)]
        p1, p2 = self._patch()
        with p1, p2:
            executor.apply(rules)
        self.assertEqual(rules[0].value, 1)

    # ------------------------------------------------------------------ #
    #  _mutate — direct tests
    # ------------------------------------------------------------------ #
    def test_mutate_does_not_modify_original(self):
        executor = MutationExecutor(
            literal_mutations=[_IncrementLiteral()],
            operator_mutations=[_ToggleOperator()],
            mutation_p=1.0,
        )
        rule = Literal(value=0)
        p1, p2 = self._patch()
        with p1, p2:
            result = executor._mutate(rule, 1)
        self.assertEqual(rule.value, 0)
        self.assertEqual(result.value, 1)

    def test_mutate_multiple_mutations(self):
        executor = MutationExecutor(
            literal_mutations=[_IncrementLiteral()],
            operator_mutations=[_ToggleOperator()],
            mutation_p=1.0,
        )
        rule = Literal(value=0)
        p1, p2 = self._patch()
        with p1, p2:
            result = executor._mutate(rule, 3)
        self.assertEqual(result.value, 3)

    def test_mutate_returns_original_on_total_failure(self):
        executor = MutationExecutor(
            literal_mutations=[_FailingMutation()],
            operator_mutations=[_ToggleOperator()],
            mutation_p=1.0,
        )
        rule = Literal(value=7)
        p1, p2 = self._patch()
        with p1, p2:
            result = executor._mutate(rule, 1)
        self.assertEqual(result.value, 7)

    # ------------------------------------------------------------------ #
    #  Doctests
    # ------------------------------------------------------------------ #
    def test_doctests(self):
        result = doctest.testmod(hgp_lib.mutations.mutation_executor, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
