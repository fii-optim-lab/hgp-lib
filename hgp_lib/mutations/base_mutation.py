from abc import ABC, abstractmethod

from ..rules import Rule


# TODO: Add documentation and doctests and tests


class Mutation(ABC):
    def __init__(self, is_literal_mutation: bool, is_operator_mutation: bool):
        if not isinstance(is_literal_mutation, bool):
            raise TypeError(f"is_literal_mutation must be a bool, is '{type(is_literal_mutation)}'")
        if not isinstance(is_operator_mutation, bool):
            raise TypeError(f"is_operator_mutation must be a bool, is '{type(is_operator_mutation)}'")
        if not is_literal_mutation and not is_operator_mutation:
            raise ValueError(f"A Mutation must be at least either a literal mutation, or an operator mutation. "
                             f"{type(self)} is neither.")

        self.is_literal_mutation: bool = is_literal_mutation
        self.is_operator_mutation: bool = is_operator_mutation

    @abstractmethod
    def apply(self, rule: Rule):
        pass
