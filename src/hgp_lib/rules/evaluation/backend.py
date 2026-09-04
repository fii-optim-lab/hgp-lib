from typing import Protocol

from hgp_lib.rules import Rule


class EvaluationBackend(Protocol):
    def prepare_data(self, data):
        ...

    def evaluate(self, rule: Rule, data):
        ...