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


def create_standard_literal_mutations(
    num_literals: int, operator_types: Sequence[Type[Rule]] = (Or, And)
) -> Tuple[Mutation, ...]:
    """
    Creates a standard set of literal-level mutations commonly used in rule evolution.

    Args:
        num_literals (int):
            Total number of available literal values. Must be greater than `1`.
        operator_types (Sequence[Type[Rule]]):
            Sequence of operator classes (e.g., `(Or, And)`) used by `PromoteLiteral`. Default: `(Or, And)`.

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

    The returned tuple includes:

    Args:
        num_literals (int):
            Total number of available literal values. Must be greater than `1`.
        operator_types (Sequence[Type[Rule]]):
            Sequence of operator classes (e.g., `(Or, And)`) used by `ReplaceOperator`. Default: `(Or, And)`.

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


def create_default_mutation_executor(
    num_literals: int, mutation_p: float, check_valid: Callable[[Rule], bool] | None
) -> MutationExecutor:
    literal_mutation_fn = create_standard_literal_mutations
    operator_mutation_fn = create_standard_operator_mutations
    num_tries = 1
    return MutationExecutor(
        literal_mutations=literal_mutation_fn(num_literals),
        operator_mutations=operator_mutation_fn(num_literals),
        mutation_p=mutation_p,
        check_valid=check_valid,
        num_tries=num_tries,
    )


create_mutation_executor_fn_type = Callable[
    [int, float, Callable[[Rule], bool] | None], MutationExecutor
]
