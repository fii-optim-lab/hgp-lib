from typing import Sequence, Type

from ..rules import Rule


class MutationError(Exception):
    # Placeholder class used to signify that a mutation could not be executed because it would violate constraints
    pass


def validate_num_literals(num_literals: int):
    if not isinstance(num_literals, int):
        raise TypeError(
            f"Number of literals must be an integer, is '{type(num_literals)}'"
        )
    if num_literals <= 0:
        raise ValueError(
            f"Number of literals must be a positive integer, is '{num_literals}'"
        )


def validate_operator_types(operator_types: Sequence[Type[Rule]]):
    if not isinstance(operator_types, Sequence):
        raise TypeError(
            f"operator_types must be a Sequence, is '{type(operator_types)}'"
        )
    if len(operator_types) < 2:
        raise ValueError("operator_types must have at least two operator types")
    for operator_type in operator_types:
        if not issubclass(operator_type, Rule):
            raise TypeError(
                f"All operator types must be subclassing Rule. Found '{type(operator_type)}'"
            )
