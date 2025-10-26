import numpy as np

from .rules import Rule


class Literal(Rule):
    def evaluate(self, data):
        return np.logical_not(data[:, self.value]) if self.negated else data[:, self.value]

    def __str__(self):
        return f"~{self.value}" if self.negated else f"{self.value}"
