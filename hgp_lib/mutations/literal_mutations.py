import random
from typing import Sequence, Type, Tuple

import numpy as np

from .base_mutation import Mutation
from .utils import MutationError, validate_num_literals, validate_operator_types
from ..rules import Rule, Or, And, Literal


class DeleteMutation(Mutation):
    def __init__(self):
        super().__init__(is_literal_mutation=True, is_operator_mutation=True)

    def apply(self, rule: Rule):
        parent = rule.parent
        # TODO: @? Should we keep 2 as the limit? Maybe we can let operators have 1 subrule?
        if parent is None or len(parent.subrules) == 2:
            raise MutationError()
        for i in range(len(parent.subrules)):
            if parent.subrules[i] is rule:  # We use reference checking because it is faster
                del parent.subrules[i]
                return
        raise RuntimeError("Unreachable code")


class NegateMutation(Mutation):
    def __init__(self):
        super().__init__(is_literal_mutation=True, is_operator_mutation=True)

    def apply(self, rule: Rule):
        rule.negated = not rule.negated


class ReplaceLiteral(Mutation):
    def __init__(self, num_literals: int):
        super().__init__(is_literal_mutation=True, is_operator_mutation=False)

        validate_num_literals(num_literals)
        self.num_literals = num_literals

    def apply(self, rule: Rule):
        new_value = np.random.randint(self.num_literals)
        if new_value == rule.value:
            new_value = (new_value + 1) % self.num_literals
        rule.value = new_value


class PromoteLiteral(Mutation):
    def __init__(self, num_literals: int, operator_types: Sequence[Type[Rule]] = (Or, And)):
        super().__init__(is_literal_mutation=True, is_operator_mutation=False)

        validate_num_literals(num_literals)
        validate_operator_types(operator_types)

        self.num_literals = num_literals
        self.operator_types: Tuple[Type[Rule]] = tuple(operator_types)

    def apply(self, rule: Rule):
        rule.__class__ = random.choice(self.operator_types)  # Efficient class change
        negated = np.random.rand(2) < 0.5  # Creating negated flag for the new operator and the new literal
        new_value = np.random.randint(self.num_literals)  # Creating value for new literal
        if new_value == rule.value:
            new_value = (new_value + 1) % self.num_literals
        rule.subrules = [
            Literal(None, rule, rule.value, rule.negated),  # Old literal
            Literal(None, rule, new_value, negated[0]),  # New literal
        ]
        rule.negated = negated[1]  # Operator negated flag
        rule.value = None  # Removing the value from the new operator


def create_standard_literal_mutations(num_literals: int, operator_types: Sequence[Type[Rule]] = (Or, And)) \
        -> Tuple[Mutation, ...]:
    return (
        DeleteMutation(),
        NegateMutation(),
        ReplaceLiteral(num_literals),
        PromoteLiteral(num_literals, operator_types),
    )
