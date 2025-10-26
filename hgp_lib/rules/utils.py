from typing import Type

from .rules import Rule
from .literals import Literal


def is_operator(op: Rule):
    return isinstance(op, Rule) and not isinstance(op, Literal)


def is_operator_type(t: Type[Rule]):
    return isinstance(t, type) and issubclass(t, Rule) and not issubclass(t, Literal)
