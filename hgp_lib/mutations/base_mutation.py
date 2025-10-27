from abc import ABC, abstractmethod

from ..rules import Rule


# TODO: Add documentation and doctests and tests


class Mutation(ABC):
    def __init__(self):
        self.is_literal_mutation: bool = False
        self.is_operator_mutation: bool = False

    @abstractmethod
    def apply(self, rule: Rule) -> Rule:
        pass
