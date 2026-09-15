from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ...rules import Rule
from ..backend import EvaluationBackend
from ..context import EvaluationContext
from .evaluator import predict_rule
from .population import score_population
from .scorers import prepare_numpy_scorer


class NumpyBackend(EvaluationBackend):
    def __init__(
        self,
        *,
        low_memory: bool = False,
        batched: bool = False,
    ) -> None:
        self.low_memory = low_memory
        self.batched = batched

    def prepare(
        self,
        data: Any,
        labels: Any,
        score_fn: Callable | None,
        *,
        optimize_scorer: bool,
    ) -> EvaluationContext:
        raise NotImplementedError

    def predict(
        self,
        rule: Rule,
        data: np.ndarray,
    ) -> np.ndarray:
        return predict_rule(rule, data)

    def score_population(
        self,
        population: Sequence[Rule],
        context: EvaluationContext,
    ) -> np.ndarray:
        return score_population(
            population,
            context,
            low_memory=self.low_memory,
            batched=self.batched,
        )