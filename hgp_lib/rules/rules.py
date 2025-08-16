# Reimplementation based on https://github.com/fidelity/boolxai/blob/main/boolxai/rules/rule.py
from abc import ABC, abstractmethod
from typing import List, Optional


class Rule(ABC):
    __slots__ = ("subrules", "parent", "value", "negated")

    def __init__(
            self,
            subrules: Optional[List["Rule"]] = None,
            parent: Optional["Rule"] = None,
            value: Optional[int] = None,
            negated: bool = False
    ):
        self.subrules = [] if subrules is None else [s.copy(self) for s in subrules]
        self.parent = parent
        self.value = value
        self.negated = negated

    def flatten(self):
        result = [self]
        for subrule in self.subrules:
            result.extend(subrule.flatten())
        return result

    def __len__(self):
        return 1 + sum([len(s) for s in self.subrules])

    def __str__(self):
        return f"{type(self).__name__}({', '.join(str(s) for s in self.subrules)})"

    def __repr__(self):
        return self.__str__()

    def copy(self, parent):
        return self.__class__(self.subrules, parent or self.parent, self.value, self.negated)

    @abstractmethod
    def evaluate(self, data):
        pass
