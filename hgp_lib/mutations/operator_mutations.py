import random
from typing import Tuple, Type, Sequence

from .base_mutation import Mutation
from .literal_mutations import DeleteMutation, NegateMutation
from .utils import MutationError
from ..rules import Rule, Or, And, Literal


class RemoveIntermediateOperator(Mutation):
    def __init__(self):
        super().__init__(is_literal_mutation=False, is_operator_mutation=True)

    def apply(self, rule: Rule):
        # It moves the children of the operator to the parent of the operator
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
    def __init__(self, operator_types: Sequence[Type[Rule]] = (Or, And)):
        super().__init__(is_literal_mutation=False, is_operator_mutation=True)

        if not isinstance(operator_types, Sequence):
            raise TypeError(f"operator_types must be a Sequence, is '{type(operator_types)}'")
        if len(operator_types) < 2:
            raise ValueError(f"operator_types must have at least two operator types")
        for operator_type in operator_types:
            if not issubclass(operator_type, Rule):
                raise TypeError(f"All operator types must be subclassing Rule. Found '{type(operator_type)}'")

        self.operator_types: Tuple[Type[Rule]] = tuple(operator_types)

    def apply(self, rule: Rule):
        other_operators = [operator_type for operator_type in self.operator_types if not isinstance(rule, operator_type)]
        rule.__class__ = random.choice(other_operators)


class AddLiteral(Mutation):
    def __init__(self, num_literals: int):
        super().__init__(is_literal_mutation=False, is_operator_mutation=True)

        if not isinstance(num_literals, int):
            raise TypeError(f"Number of literals must be an integer, is '{type(num_literals)}'")
        if num_literals <= 0:
            raise ValueError(f"Number of literals must be a positive integer, is '{num_literals}'")

        self.available_literals = set(range(num_literals))

    def apply(self, rule: Rule):
        # TODO: Check performance. Maybe a better implementation is needed.
        available_literals = tuple(self.available_literals.difference([s.value for s in rule.subrules if s.value]))
        if len(available_literals) == 0:
            raise MutationError()
        rule.subrules.append(
            Literal(None, rule, random.choice(available_literals), random.random() < 0.5)
        )


def create_standard_operator_mutations(num_literals: int, operator_types: Sequence[Type[Rule]] = (Or, And)) \
        -> Tuple[Mutation, ...]:
    return (
        DeleteMutation(),
        NegateMutation(),
        RemoveIntermediateOperator(),
        ReplaceOperator(operator_types),
        AddLiteral(num_literals),
    )
