from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..rules import Rule
from .context import EvaluationContext


class EvaluationBackend(ABC):
    @abstractmethod
    def prepare(
        self,
        data: Any,
        labels: Any,
        score_fn: Callable | None,
        *,
        optimize_scorer: bool,
    ) -> EvaluationContext:
        pass

    @abstractmethod
    def predict(
        self,
        rule: Rule,
        data: Any,
    ) -> Any:
        pass

    @abstractmethod
    def score_population(
        self,
        population: Sequence[Rule],
        context: EvaluationContext,
    ) -> np.ndarray:
        pass

    @staticmethod
    def to_numpy(values: Any) -> np.ndarray:
        return np.asarray(values)