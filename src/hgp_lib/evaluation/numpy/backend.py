from collections.abc import Callable

import numpy as np

from ...rules import Rule
from .. import EvaluationBackend, EvaluationContext


class NumpyBackend(EvaluationBackend):
    def prepare_evaluation_context(self, data: np.ndarray, labels: np.ndarray, score_fn: Callable | None,
                                   optimize_scorer: bool) -> EvaluationContext:
        if optimize_scorer:
            data, labels, score_fn = ...
        return EvaluationContext(data, labels, score_fn=score_fn)

    def evaluate_population(self, population: list[Rule], evaluation: EvaluationContext) -> np.ndarray:
        pass

    def evaluate_rule(self, rule: Rule, data: np.ndarray) -> np.ndarray:
        pass

