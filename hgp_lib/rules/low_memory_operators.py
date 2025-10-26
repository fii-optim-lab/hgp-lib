import numpy as np

from .rules import Rule


# TODO: Add some performance tests comparing operator implementations
class And(Rule):
    def evaluate(self, data):
        rez = self.subrules[0].evaluate(data)
        if self.subrules[0].value is not None and not self.subrules[0].negated:
            rez = rez.copy()
        for subrule in self.subrules[1:]:
            rez &= subrule.evaluate(data)
        if self.negated:
            rez = np.logical_not(rez, out=rez)
        return rez


class Or(Rule):
    def evaluate(self, data):
        rez = self.subrules[0].evaluate(data)
        if self.subrules[0].value is not None and not self.subrules[0].negated:
            rez = rez.copy()
        for subrule in self.subrules[1:]:
            rez |= subrule.evaluate(data)
        if self.negated:
            rez = np.logical_not(rez, out=rez)
        return rez
