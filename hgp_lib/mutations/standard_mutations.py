import random
from typing import Type, Tuple, Iterable

import numpy as np

from hgp_lib.mutations.mutation_executor import MutationError
from hgp_lib.rules import Rule, Literal, Or, And


# TODO: Add tests for all mutations

# TODO: Add code and checks such that we do not depend on our Rules. And someone else can use use without using the
#  rules


# Operator and Literal mutations
def delete(gene: Rule) -> int:
    # Can be applied to both literal and operators
    # Returns the length of the subtree of the current gene.
    parent = gene.parent
    if parent is None or len(parent.subrules) < 3:
        raise MutationError("Can't delete root node or subrules from operators with less than 3 subrules")
    for i in range(len(parent.subrules)):
        if parent.subrules[i] is gene:  # We use reference checking because it is faster
            del parent.subrules[i]
            return len(gene) - 1
    raise RuntimeError("Unreachable code")


def negate(gene: Rule) -> int:
    gene.negated = not gene.negated
    return 0


# Operator mutations
def remove_intermediate_operator(operator: Rule) -> int:
    # This mutation can be applied only to operators
    # It moves the children of the operator to the parent of the operator
    parent = operator.parent
    if parent is None:
        raise MutationError("Can't remove root node")
    for s in operator.subrules:
        s.parent = parent
    parent.subrules += operator.subrules
    for i in range(len(parent.subrules)):
        if parent.subrules[i] is operator:
            del parent.subrules[i]
            return 0
    raise RuntimeError("Unreachable code")


class replace_operator:
    def __init__(self, operator_types: Tuple[Type[Rule]] = (Or, And)):
        assert isinstance(operator_types, Iterable) and len(operator_types) >= 2 and all([
            issubclass(x, Rule) for x in operator_types
        ]), f"Operator types must be a list with at least two operator types, subclassing Rule. Is {operator_types}"

        self.operator_types = operator_types

    def __call__(self, operator: Rule) -> int:
        other_operators = [operator_type for operator_type in self.operator_types if
                           not isinstance(operator, operator_type)]
        operator.__class__ = random.choice(other_operators)
        return 0


class add_literal:
    def __init__(self, num_literals: int):
        assert isinstance(num_literals, int) and num_literals > 0, \
            f"Number of literals must be a positive integer, is {num_literals}"

        self.available_literals = set(range(num_literals))

    def __call__(self, operator: Rule) -> int:
        used_literals = [s.value for s in operator.subrules if s.value]
        try:
            operator.subrules.append(
                Literal(
                    None,  # subrules
                    operator,  # parent
                    random.choice(tuple(self.available_literals.difference(used_literals))),  # value
                    np.random.rand() < 0.5  # negated
                )
            )
        except IndexError:
            raise MutationError("All available literals were used")
        return 0


# Literal mutations
class replace_literal:
    def __init__(self, num_literals: int):
        assert isinstance(num_literals, int) and num_literals >= 2, \
            f"Number of literals must be a positive integer greater or equal to 2, is {num_literals}"

        self.num_literals = num_literals

    def __call__(self, literal: Rule) -> int:
        new_value = np.random.randint(self.num_literals)
        if new_value == literal.value:
            new_value = (new_value + 1) % self.num_literals
        literal.value = new_value
        return 0


class promote_literal:
    def __init__(self, num_literals: int, operator_types: Tuple[Type[Rule]] = (Or, And)):
        assert isinstance(num_literals, int) and num_literals >= 2, \
            f"Number of literals must be a positive integer greater or equal to 2, is {num_literals}"
        assert isinstance(operator_types, Iterable) and len(operator_types) >= 2 and all([
            issubclass(x, Rule) for x in operator_types
        ]), f"Operator types must be a list with at least two operator types, subclassing Rule. Is {operator_types}"

        self.num_literals = num_literals
        self.operator_types = operator_types

    def __call__(self, literal: Rule) -> int:
        literal.__class__ = random.choice(self.operator_types)  # Efficient class change
        negated = np.random.rand(2) < 0.5  # Creating negated flag for the new operator and the new literal
        new_value = np.random.randint(self.num_literals)  # Creating value for new literal
        if new_value == literal.value:
            new_value = (new_value + 1) % self.num_literals
        literal.subrules = [
            Literal(None, literal, literal.value, literal.negated),  # Old literal
            Literal(None, literal, new_value, negated[0]),  # New literal
        ]
        literal.negated = negated[1]  # Operator negated flag
        literal.value = None  # Removing the value from the new operator
        return 0
