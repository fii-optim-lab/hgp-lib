from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


ScoreFunction = Callable[[Any, Any], float]


class PreparedScorer(ABC):
    @abstractmethod
    def score(self, predictions: Any) -> float:
        """Score predictions for the prepared labels."""

    def score_population(self, predictions: Any) -> Any:
        """Optionally score multiple prediction vectors."""
        raise NotImplementedError()


class CallableScorer(PreparedScorer):
    def __init__(
        self,
        labels: Any,
        score_fn: ScoreFunction,
    ):
        self.labels = labels
        self.score_fn = score_fn

    def score(self, predictions: Any) -> float:
        return self.score_fn(self.labels, predictions)


def prepare_scorer(
    labels: Any,
    score_fn: ScoreFunction | None,
    *,
    backend: str,
    sample_weight: Any | None = None,
    optimize: bool = True,
) -> PreparedScorer:
    raise NotImplementedError()