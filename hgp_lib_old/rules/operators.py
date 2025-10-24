from typing import Type

import numpy as np

from hgp_lib.rules import Literal
from hgp_lib.rules.rules import Rule


def is_operator(op: Rule):
    return isinstance(op, Rule) and not isinstance(op, Literal)


def is_operator_type(t: Type[Rule]):
    return isinstance(t, type) and issubclass(t, Rule) and not issubclass(t, Literal)


class And(Rule):
    def evaluate(self, data):
        cols = []
        neg_mask = []
        sub_operators = []
        for s in self.subrules:
            if s.value is not None:  # We have a literal
                cols.append(s.value)
                neg_mask.append(s.negated)
            else:  # We have an operator
                sub_operators.append(s)

        if cols:  # Hot branch for literals
            mask = (data[:, cols] ^ np.array(neg_mask, dtype=np.bool)).all(axis=1)  # One-liner for all literals
            # Updating with operators
            for s in sub_operators:
                mask &= s.evaluate(data)
        else:  # Hot branch for no literals
            mask = sub_operators[0].evaluate(data)  # Create an initial matrix
            for s in sub_operators[1:]:  # Updating with the rest of the operators
                mask &= s.evaluate(data)

        if self.negated:
            mask = np.logical_not(mask, out=mask)
        return mask


class Or(Rule):
    def evaluate(self, data):
        cols = []
        neg_mask = []
        sub_operators = []
        for s in self.subrules:
            if s.value is not None:  # We have a literal
                cols.append(s.value)
                neg_mask.append(s.negated)
            else:  # We have an operator
                sub_operators.append(s)

        if cols:  # Hot branch for literals
            mask = (data[:, cols] ^ np.array(neg_mask, dtype=np.bool)).any(axis=1)  # One-liner for all literals
            # Updating with operators
            for s in sub_operators:
                mask |= s.evaluate(data)
        else:  # Hot branch for no literals
            mask = sub_operators[0].evaluate(data)  # Create an initial matrix
            for s in sub_operators[1:]:  # Updating with the rest of the operators
                mask |= s.evaluate(data)

        if self.negated:
            mask = np.logical_not(mask, out=mask)
        return mask

# TODO: Add higher level operators from Boolxai
