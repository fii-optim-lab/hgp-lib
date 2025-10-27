import random
from typing import Tuple, Type, Sequence

from .base_mutation import Mutation
from .literal_mutations import DeleteMutation, NegateMutation
from .utils import MutationError, validate_num_literals, validate_operator_types
from ..rules import Rule, Or, And, Literal


class RemoveIntermediateOperator(Mutation):
    """
    The `RemoveIntermediateOperator` mutation removes an intermediate operator node (e.g., `And`, `Or`) and promotes its
    subrules to the operator’s parent, flattening the rule tree structure.

    Attributes:
        is_literal_mutation (bool): `False`.
        is_operator_mutation (bool): `True`.

    Notes:
        - The mutation raises a `MutationError` if the operator has no parent (i.e., it is the root of the tree).
        - All child subrules of the removed operator are reattached directly to the parent.
        - The parent’s subrule list is updated inplace, preserving the relative order of existing subrules.

    Examples:
        >>> from hgp_lib.mutations import RemoveIntermediateOperator
        >>> from hgp_lib.rules import And, Or, Literal
        >>> rule = And([
        ...     Literal(value=0),
        ...     Or([
        ...         Literal(value=1),
        ...         Literal(value=2),
        ...     ]),
        ...     Literal(value=3)
        ... ])
        >>> mutation = RemoveIntermediateOperator()
        >>> mutation.apply(rule.subrules[1])
        >>> rule
        And(0, 3, 1, 2)
    """

    def __init__(self):
        super().__init__(is_literal_mutation=False, is_operator_mutation=True)

    def apply(self, rule: Rule):
        """
        Applies an inplace structural mutation that removes the specified operator and attaches its subrules directly
        to the parent operator.

        Args:
            rule (Rule):
                The operator rule to remove.

        Raises:
            MutationError:
                If the operator has no parent (i.e., it is the root node).
            RuntimeError:
                If the target operator is not found within its parent’s subrule list, which should never occur during
                normal operation.

        Examples:
            >>> from hgp_lib.mutations import RemoveIntermediateOperator
            >>> from hgp_lib.rules import Or, And, Literal
            >>> rule = Or([
            ...     Literal(value=1),
            ...     And([
            ...         Literal(value=2),
            ...         Literal(value=3),
            ...     ]),
            ...     Literal(value=4)
            ... ])
            >>> mutation = RemoveIntermediateOperator()
            >>> mutation.apply(rule.subrules[1])
            >>> rule
            Or(1, 4, 2, 3)
        """
        parent = rule.parent
        if parent is None:
            raise MutationError()
        for s in rule.subrules:
            s.parent = parent
        parent.subrules += rule.subrules
        for i in range(len(parent.subrules)):
            if parent.subrules[i] is rule:
                del parent.subrules[i]
                return
        raise RuntimeError("Unreachable code")


class ReplaceOperator(Mutation):
    """
    The `ReplaceOperator` mutation replaces an operator node (e.g., `And`, `Or`) with another operator type chosen
    randomly from the provided set of operator classes. The transformation occurs inplace, preserving the subrules
    and other attributes of the node.

    Attributes:
        is_literal_mutation (bool): `False`.
        is_operator_mutation (bool): `True`.
        operator_types (Tuple[Type[Rule]]): Tuple of operator classes that can replace one another (e.g., `(Or, And)`).

    Notes:
        - The mutation has no effect on literal nodes.
        - The operator type is switched inplace by directly reassigning the node’s `__class__` attribute.
        - The replacement operator is always different from the current operator type.

    Examples:
        >>> from hgp_lib.mutations import ReplaceOperator
        >>> from hgp_lib.rules import And, Or, Literal
        >>> rule = And([Literal(value=0), Literal(value=1)])
        >>> mutation = ReplaceOperator(operator_types=(Or, And))
        >>> mutation.apply(rule)
        >>> rule
        Or(0, 1)
    """

    def __init__(self, operator_types: Sequence[Type[Rule]] = (Or, And)):
        super().__init__(is_literal_mutation=False, is_operator_mutation=True)
        validate_operator_types(operator_types)
        self.operator_types: Tuple[Type[Rule]] = tuple(operator_types)

    def apply(self, rule: Rule):
        """
        Applies an inplace operator replacement mutation to the given `Rule`, changing its type to a different operator
        from the available set.

        Args:
            rule (Rule):
                The operator rule to replace. Must be an instance of one of the `operator_types`.

        Examples:
            >>> from hgp_lib.mutations import ReplaceOperator
            >>> from hgp_lib.rules import And, Or, Literal
            >>> rule = Or([Literal(value=2), Literal(value=3)])
            >>> mutation = ReplaceOperator()
            >>> mutation.apply(rule)
            >>> rule
            And(2, 3)
        """
        other_operators = [operator_type for operator_type in self.operator_types if
                           not isinstance(rule, operator_type)]
        rule.__class__ = random.choice(other_operators)


class AddLiteral(Mutation):
    """
    The `AddLiteral` mutation adds a new literal subrule to an operator node (e.g., `And`, `Or`).
    The new literal is chosen randomly from the available literal space, ensuring it does not duplicate
    an existing literal value already present in the operator.

    Attributes:
        is_literal_mutation (bool): `False`.
        is_operator_mutation (bool): `True`.
        available_literals (Set[int]): Set of all possible literal indices from which new literals are sampled.

    Notes:
        - The mutation raises a `MutationError` if all possible literals are already present under the operator.
        - The added literal’s negation flag is determined randomly with equal probability.
        - The mutation operates inplace, modifying the operator’s list of subrules directly.

    Examples:
        >>> from hgp_lib.mutations import AddLiteral
        >>> from hgp_lib.rules import And, Literal
        >>> rule = And([Literal(value=0), Literal(value=1)])
        >>> mutation = AddLiteral(num_literals=3)
        >>> mutation.apply(rule)
        >>> rule.subrules[2].negated=True  # Setting negated to have deterministic output
        >>> rule
        And(0, 1, ~2)
    """

    def __init__(self, num_literals: int):
        super().__init__(is_literal_mutation=False, is_operator_mutation=True)

        validate_num_literals(num_literals)

        self.available_literals = set(range(num_literals))

    def apply(self, rule: Rule):
        """
        Adds a new literal subrule to the given operator node. The new literal’s value is selected randomly from the
        remaining available literals not already present in the operator.

        Args:
            rule (Rule):
                The operator rule to which the new literal will be added.

        Raises:
            MutationError:
                If no new literal values are available for addition.

        Examples:
            >>> from hgp_lib.mutations import AddLiteral
            >>> from hgp_lib.rules import Or, Literal
            >>> mutation = AddLiteral(num_literals=3)
            >>> rule = Or([Literal(value=0), Literal(value=1)])
            >>> mutation.apply(rule)
            >>> rule.subrules[2].negated=False  # Setting negated to have deterministic output
            >>> rule.subrules[2]
            2
            >>> rule
            Or(0, 1, 2)
        """
        # TODO: Check performance. Maybe a better implementation is needed.
        available_literals = tuple(
            self.available_literals.difference([s.value for s in rule.subrules if s.value is not None]))
        if len(available_literals) == 0:
            raise MutationError()
        rule.subrules.append(
            Literal(None, rule, random.choice(available_literals), random.random() < 0.5)
        )


def create_standard_operator_mutations(num_literals: int, operator_types: Sequence[Type[Rule]] = (Or, And)) \
        -> Tuple[Mutation, ...]:
    """
    Creates a standard set of operator-level mutations used for structural modifications of rule trees.

    The returned tuple includes:

    Args:
        num_literals (int):
            Total number of available literal values. Must be greater than 1.
        operator_types (Sequence[Type[Rule]]):
            Sequence of operator classes (e.g., `(Or, And)`) used by `ReplaceOperator`. Default: `(Or, And)`.

    Returns:
        Tuple[Mutation, ...]:
            A tuple of initialized operator-level mutation instances. The tuple includes:
            1. `DeleteMutation()` — removes a rule from its parent operator.
            2. `NegateMutation()` — toggles the negation flag of an operator.
            3. `RemoveIntermediateOperator()` — removes an intermediate operator and promotes its children.
            4. `ReplaceOperator(operator_types)` — replaces an operator with another type (e.g., `And` with `Or`).
            5. `AddLiteral(num_literals)` — adds a new literal subrule to the operator.


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
