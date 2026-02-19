from typing import Callable, Sequence, Tuple, Type

from .base_mutation import Mutation
from .literal_mutations import (
    DeleteMutation,
    NegateMutation,
    PromoteLiteral,
    ReplaceLiteral,
)
from .operator_mutations import AddLiteral, RemoveIntermediateOperator, ReplaceOperator
from ..rules import And, Or, Rule
from .mutation_executor import MutationExecutor
from ..utils.validation import check_isinstance


def create_standard_literal_mutations(
    num_literals: int, operator_types: Sequence[Type[Rule]] = (Or, And)
) -> Tuple[Mutation, ...]:
    """
    Creates a standard set of literal-level mutations commonly used in rule evolution.

    Args:
        num_literals (int):
            Total number of available literal values. Must be greater than `1`.
        operator_types (Sequence[Type[Rule]]):
            Sequence of operator classes (e.g., `(Or, And)`) used by `PromoteLiteral`.
            Default: `(Or, And)`.

    Returns:
        Tuple[Mutation, ...]:
            A tuple of initialized mutation instances for literals. The tuple includes:
            1. `DeleteMutation()` - removes a rule from its parent operator.
            2. `NegateMutation()` - toggles the negation flag of a rule.
            3. `ReplaceLiteral(num_literals)` - replaces a literal's value with a different random one.
            4. `PromoteLiteral(num_literals, operator_types)` - converts a literal into an operator with two literals.

    Examples:
        >>> from hgp_lib.mutations import create_standard_literal_mutations
        >>> from hgp_lib.rules import And, Or
        >>> mutations = create_standard_literal_mutations(num_literals=4, operator_types=(Or, And))
        >>> [type(mutation).__name__ for mutation in mutations]
        ['DeleteMutation', 'NegateMutation', 'ReplaceLiteral', 'PromoteLiteral']
    """
    return (
        DeleteMutation(),
        NegateMutation(),
        ReplaceLiteral(num_literals),
        PromoteLiteral(num_literals, operator_types),
    )


def create_standard_operator_mutations(
    num_literals: int, operator_types: Sequence[Type[Rule]] = (Or, And)
) -> Tuple[Mutation, ...]:
    """
    Creates a standard set of operator-level mutations used for structural modifications of rule trees.

    Args:
        num_literals (int):
            Total number of available literal values. Must be greater than `1`.
        operator_types (Sequence[Type[Rule]]):
            Sequence of operator classes (e.g., `(Or, And)`) used by `ReplaceOperator`.
            Default: `(Or, And)`.

    Returns:
        Tuple[Mutation, ...]:
            A tuple of initialized operator-level mutation instances. The tuple includes:
            1. `DeleteMutation()` - removes a rule from its parent operator.
            2. `NegateMutation()` - toggles the negation flag of an operator.
            3. `RemoveIntermediateOperator()` - removes an intermediate operator and promotes its children.
            4. `ReplaceOperator(operator_types)` - replaces an operator with another type (e.g., `And` with `Or`).
            5. `AddLiteral(num_literals)` - adds a new literal subrule to the operator.

    Examples:
        >>> from hgp_lib.mutations import create_standard_operator_mutations
        >>> from hgp_lib.rules import And, Or
        >>> mutations = create_standard_operator_mutations(num_literals=4, operator_types=(Or, And))
        >>> [type(mutation).__name__ for mutation in mutations]
        ['DeleteMutation', 'NegateMutation', 'RemoveIntermediateOperator', 'ReplaceOperator', 'AddLiteral']
    """
    return (
        DeleteMutation(),
        NegateMutation(),
        RemoveIntermediateOperator(),
        ReplaceOperator(operator_types),
        AddLiteral(num_literals),
    )


class MutationExecutorFactory:
    """
    Factory for creating `MutationExecutor` instances.

    Stores configuration-time parameters (`mutation_p`, `num_tries`) and defers
    data-dependent construction to `create`. Override `create_literal_mutations`
    and/or `create_operator_mutations` to customise which mutations are used.

    Attributes:
        mutation_p (float): Per-node mutation probability. Default: `0.1`.
        num_tries (int): Maximum number of attempts per mutation node. Must be `1`
            when no `check_valid` is provided to `create`. Default: `1`.
        operator_p (float): Probability of selecting an operator node (vs. a literal)
            when choosing a mutation point in the rule tree. Must be in [0.0, 1.0].
            Default: `0.9`.

    Examples:
        >>> from hgp_lib.mutations import MutationExecutorFactory
        >>> factory = MutationExecutorFactory(mutation_p=0.05)
        >>> factory.mutation_p
        0.05
        >>> factory.num_tries
        1

        Subclass to use custom mutations:

        >>> from hgp_lib.mutations import MutationExecutorFactory, NegateMutation
        >>> class NegateOnlyFactory(MutationExecutorFactory):
        ...     def create_literal_mutations(self, num_literals):
        ...         return (NegateMutation(),)
        ...     def create_operator_mutations(self, num_literals):
        ...         return (NegateMutation(),)
        >>> factory = NegateOnlyFactory(mutation_p=1.0)
        >>> executor = factory.create(num_literals=4)
        >>> len(executor.literal_mutations)
        1
    """

    def __init__(
        self, mutation_p: float = 0.1, num_tries: int = 1, operator_p: float = 0.9
    ):
        check_isinstance(mutation_p, float)
        if mutation_p < 0.0 or mutation_p > 1.0:
            raise ValueError(
                f"mutation_p must be between 0.0 and 1.0, got {mutation_p}"
            )
        check_isinstance(num_tries, int)
        if num_tries < 1:
            raise ValueError(f"num_tries must be at least 1, got {num_tries}")
        check_isinstance(operator_p, float)
        if operator_p < 0.0 or operator_p > 1.0:
            raise ValueError(
                f"operator_p must be between 0.0 and 1.0, got {operator_p}"
            )
        self.mutation_p = mutation_p
        self.num_tries = num_tries
        self.operator_p = operator_p

    def create_literal_mutations(self, num_literals: int) -> Tuple[Mutation, ...]:
        """
        Create the set of literal-level mutations.

        Override this method to use custom literal mutations. The default delegates
        to `create_standard_literal_mutations(num_literals)`.

        Args:
            num_literals (int): Total number of available literal values.

        Returns:
            Tuple[Mutation, ...]: Literal mutations for the executor.
        """
        return create_standard_literal_mutations(num_literals)

    def create_operator_mutations(self, num_literals: int) -> Tuple[Mutation, ...]:
        """
        Create the set of operator-level mutations.

        Override this method to use custom operator mutations. The default delegates
        to `create_standard_operator_mutations(num_literals)`.

        Args:
            num_literals (int): Total number of available literal values.

        Returns:
            Tuple[Mutation, ...]: Operator mutations for the executor.
        """
        return create_standard_operator_mutations(num_literals)

    def create(
        self,
        num_literals: int,
        check_valid: Callable[[Rule], bool] | None = None,
    ) -> MutationExecutor:
        """
        Create a `MutationExecutor` with the configured mutations and parameters.

        Args:
            num_literals (int): Total number of available literal values.
            check_valid (Callable[[Rule], bool] | None): Optional rule validator.
                Default: `None`.

        Returns:
            MutationExecutor: Configured mutation executor.
        """
        return MutationExecutor(
            literal_mutations=self.create_literal_mutations(num_literals),
            operator_mutations=self.create_operator_mutations(num_literals),
            mutation_p=self.mutation_p,
            check_valid=check_valid,
            num_tries=self.num_tries,
            operator_p=self.operator_p,
        )
