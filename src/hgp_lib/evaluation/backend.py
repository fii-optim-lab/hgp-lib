from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from ..rules import Rule
from .evaluation_context import EvaluationContext


class EvaluationBackend(ABC):
    @abstractmethod
    def prepare_evaluation_context(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        score_fn: Callable | None,
        optimize_scorer: bool,
    ) -> EvaluationContext:
        pass


    @abstractmethod
    def evaluate_population(
        self,
        population: list[Rule],
        evaluation: EvaluationContext,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate_rule(
        self,
        rule: Rule,
        data: np.ndarray,
    ) -> np.ndarray:
        pass
