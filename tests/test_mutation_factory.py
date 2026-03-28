"""Tests for MutationExecutorFactory validation and creation."""

import doctest
import unittest

import hgp_lib.mutations.mutation_factory
from hgp_lib.mutations import (
    Mutation,
    MutationExecutor,
    MutationExecutorFactory,
    NegateMutation,
)
from hgp_lib.rules import Rule


class _LiteralOnly(Mutation):
    def __init__(self):
        super().__init__(True, False)

    def apply(self, rule: Rule):
        pass


class _OperatorOnly(Mutation):
    def __init__(self):
        super().__init__(False, True)

    def apply(self, rule: Rule):
        pass


class TestMutationExecutorFactory(unittest.TestCase):
    # ------------------------------------------------------------------ #
    #  Constructor validation
    # ------------------------------------------------------------------ #
    def test_mutation_p_must_be_float(self):
        with self.assertRaises(TypeError):
            MutationExecutorFactory(mutation_p=1)

    def test_mutation_p_bounds_low(self):
        with self.assertRaises(ValueError):
            MutationExecutorFactory(mutation_p=-0.1)

    def test_mutation_p_bounds_high(self):
        with self.assertRaises(ValueError):
            MutationExecutorFactory(mutation_p=1.5)

    def test_num_tries_must_be_int(self):
        with self.assertRaises(TypeError):
            MutationExecutorFactory(num_tries=1.5)

    def test_num_tries_must_be_positive(self):
        with self.assertRaises(ValueError):
            MutationExecutorFactory(num_tries=0)

    def test_operator_p_must_be_float(self):
        with self.assertRaises(TypeError):
            MutationExecutorFactory(operator_p=1)

    def test_operator_p_bounds(self):
        with self.assertRaises(ValueError):
            MutationExecutorFactory(operator_p=-0.1)
        with self.assertRaises(ValueError):
            MutationExecutorFactory(operator_p=1.1)

    # ------------------------------------------------------------------ #
    #  create — validation
    # ------------------------------------------------------------------ #
    def test_num_tries_gt1_requires_check_valid(self):
        with self.assertRaises(ValueError):
            MutationExecutorFactory(num_tries=2).create(num_literals=5)

    def test_num_tries_gt1_with_check_valid_ok(self):
        executor = MutationExecutorFactory(num_tries=2).create(
            num_literals=5,
            check_valid=lambda r: True,
        )
        self.assertIsInstance(executor, MutationExecutor)

    def test_literal_mutations_empty_raises(self):
        class Empty(MutationExecutorFactory):
            def create_literal_mutations(self, num_literals):
                return tuple()

        with self.assertRaises(ValueError):
            Empty().create(5)

    def test_operator_mutations_empty_raises(self):
        class Empty(MutationExecutorFactory):
            def create_operator_mutations(self, num_literals):
                return tuple()

        with self.assertRaises(ValueError):
            Empty().create(5)

    def test_literal_mutations_not_sequence_raises(self):
        class Bad(MutationExecutorFactory):
            def create_literal_mutations(self, num_literals):
                return NegateMutation()

        with self.assertRaises(TypeError):
            Bad().create(5)

    def test_operator_mutations_not_sequence_raises(self):
        class Bad(MutationExecutorFactory):
            def create_operator_mutations(self, num_literals):
                return NegateMutation()

        with self.assertRaises(TypeError):
            Bad().create(5)

    def test_literal_mutations_wrong_flag_raises(self):
        class Bad(MutationExecutorFactory):
            def create_literal_mutations(self, num_literals, operator_types):
                return (_OperatorOnly(),)

        with self.assertRaises(TypeError):
            Bad().create(5)

    def test_operator_mutations_wrong_flag_raises(self):
        class Bad(MutationExecutorFactory):
            def create_operator_mutations(self, num_literals, operator_types):
                return (_LiteralOnly(),)

        with self.assertRaises(TypeError):
            Bad().create(5)

    # ------------------------------------------------------------------ #
    #  create — happy path
    # ------------------------------------------------------------------ #
    def test_create_returns_executor(self):
        executor = MutationExecutorFactory().create(num_literals=5)
        self.assertIsInstance(executor, MutationExecutor)

    def test_create_passes_mutation_p(self):
        executor = MutationExecutorFactory(mutation_p=0.42).create(num_literals=5)
        self.assertAlmostEqual(executor.mutation_p, 0.42)

    def test_create_passes_operator_p(self):
        executor = MutationExecutorFactory(operator_p=0.3).create(num_literals=5)
        self.assertAlmostEqual(executor.operator_p, 0.3)

    def test_create_passes_num_tries(self):
        executor = MutationExecutorFactory(num_tries=3).create(
            num_literals=5,
            check_valid=lambda r: True,
        )
        self.assertEqual(executor.num_tries, 3)

    def test_create_passes_check_valid(self):
        def fn(r):
            return True

        executor = MutationExecutorFactory().create(num_literals=5, check_valid=fn)
        self.assertIs(executor.check_valid, fn)

    # ------------------------------------------------------------------ #
    #  create_literal_mutations / create_operator_mutations defaults
    # ------------------------------------------------------------------ #
    def test_default_literal_mutations(self):
        mutations = MutationExecutorFactory().create_literal_mutations(num_literals=4)
        names = [type(m).__name__ for m in mutations]
        self.assertIn("DeleteMutation", names)
        self.assertIn("NegateMutation", names)
        self.assertIn("ReplaceLiteral", names)
        self.assertIn("PromoteLiteral", names)

    def test_default_operator_mutations(self):
        mutations = MutationExecutorFactory().create_operator_mutations(num_literals=4)
        names = [type(m).__name__ for m in mutations]
        self.assertIn("DeleteMutation", names)
        self.assertIn("NegateMutation", names)
        self.assertIn("RemoveIntermediateOperator", names)
        self.assertIn("ReplaceOperator", names)
        self.assertIn("AddLiteral", names)

    # ------------------------------------------------------------------ #
    #  Custom subclass
    # ------------------------------------------------------------------ #
    def test_custom_subclass(self):
        class NegateOnly(MutationExecutorFactory):
            def create_literal_mutations(self, num_literals):
                return (NegateMutation(),)

            def create_operator_mutations(self, num_literals):
                return (NegateMutation(),)

        executor = NegateOnly(mutation_p=1.0).create(num_literals=4)
        self.assertEqual(len(executor.literal_mutations), 1)
        self.assertEqual(len(executor.operator_mutations), 1)

    # ------------------------------------------------------------------ #
    #  Doctests
    # ------------------------------------------------------------------ #
    def test_doctests(self):
        result = doctest.testmod(hgp_lib.mutations.mutation_factory, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.mutations.base_mutation, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
