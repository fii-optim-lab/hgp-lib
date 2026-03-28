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
from ..utils.validation import (
    check_isinstance,
    validate_num_literals,
    validate_operator_types,
)


class MutationExecutorFactory:
    """
    Factory for creating `MutationExecutor` instances.

    Stores configuration-time parameters (`mutation_p`, `num_tries`) and defers
    data-dependent construction to `create`. Override `create_literal_mutations`
    and/or `create_operator_mutations` to customize which mutations are used.

    Attributes:
        mutation_p (float): Per-node mutation probability. Default: `0.1`.
        num_tries (int): Maximum number of attempts per mutation node. Must be `1`
            when no `check_valid` is provided to `create`. Default: `1`.
        operator_p (float): Probability of selecting an operator node (vs. a literal)
            when choosing a mutation point in the rule tree. Must be in [0.0, 1.0].
            Default: `0.9`.
        operator_types (Sequence[Type[Rule]]): Sequence of operator classes
            (e.g., `(Or, And)`) used by mutations. Default: `(Or, And)`.

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
        self,
        mutation_p: float = 0.1,
        num_tries: int = 1,
        operator_p: float = 0.5,
        operator_types: Sequence[Type[Rule]] = (Or, And),
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
        validate_operator_types(operator_types)

        self.mutation_p = mutation_p
        self.num_tries = num_tries
        self.operator_p = operator_p
        self.operator_types: Tuple[Type[Rule], ...] = tuple(operator_types)

    def create_literal_mutations(self, num_literals: int) -> Tuple[Mutation, ...]:
        """
        Create the set of standard literal-level mutations. Override this method to use custom literal mutations.

        Args:
            num_literals (int):
                Total number of available literal values. Must be greater than `1`.

        Returns:
            Tuple[Mutation, ...]: Literal mutations for the executor.

        Examples:
            >>> from hgp_lib.mutations import MutationExecutorFactory
            >>> from hgp_lib.rules import And, Or
            >>> mutations = MutationExecutorFactory().create_literal_mutations(num_literals=4)
            >>> [type(mutation).__name__ for mutation in mutations]
            ['DeleteMutation', 'NegateMutation', 'ReplaceLiteral', 'PromoteLiteral']
        """
        return (
            DeleteMutation(),
            NegateMutation(),
            ReplaceLiteral(num_literals),
            PromoteLiteral(num_literals, self.operator_types),
        )

    def create_operator_mutations(self, num_literals: int) -> Tuple[Mutation, ...]:
        """
        Create the set of standard operator-level mutations. Override this method to use custom operator mutations.

        Args:
            num_literals (int): Total number of available literal values.

        Returns:
            Tuple[Mutation, ...]: Operator mutations for the executor.

            Examples:
            >>> from hgp_lib.mutations import MutationExecutorFactory
            >>> from hgp_lib.rules import And, Or
            >>> mutations = MutationExecutorFactory().create_operator_mutations(num_literals=4)
            >>> [type(mutation).__name__ for mutation in mutations]
            ['DeleteMutation', 'NegateMutation', 'RemoveIntermediateOperator', 'ReplaceOperator', 'AddLiteral']
        """
        return (
            DeleteMutation(),
            NegateMutation(),
            RemoveIntermediateOperator(),
            ReplaceOperator(self.operator_types),
            AddLiteral(num_literals),
        )

    @staticmethod
    def _validate_mutations(
        literal_mutations: Tuple[Mutation, ...],
        operator_mutations: Tuple[Mutation, ...],
    ):
        check_isinstance(literal_mutations, Tuple)
        check_isinstance(operator_mutations, Tuple)

        if len(literal_mutations) == 0:
            raise ValueError("literal_mutations must be a non-empty Tuple")
        if len(operator_mutations) == 0:
            raise ValueError("operator_mutations must be a non-empty Tuple")

        for literal_mutation in literal_mutations:
            check_isinstance(literal_mutation, Mutation)
            if not literal_mutation.is_literal_mutation:
                raise TypeError(
                    f"Each literal_mutations must be a literal mutation, but '{type(literal_mutation)} is not'"
                )
        for operator_mutation in operator_mutations:
            check_isinstance(operator_mutation, Mutation)
            if not operator_mutation.is_operator_mutation:
                raise TypeError(
                    f"Each operator_mutations must be an operator mutation, but '{type(operator_mutation)} is not'"
                )

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
        validate_num_literals(num_literals)

        if check_valid is None and self.num_tries > 1:
            raise ValueError("num_tries must be 1 if check_valid is None")

        literal_mutations = self.create_literal_mutations(num_literals)
        operator_mutations = self.create_operator_mutations(num_literals)

        self._validate_mutations(
            literal_mutations,
            operator_mutations,
        )

        return MutationExecutor(
            literal_mutations=literal_mutations,
            operator_mutations=operator_mutations,
            mutation_p=self.mutation_p,
            check_valid=check_valid,
            num_tries=self.num_tries,
            operator_p=self.operator_p,
        )
