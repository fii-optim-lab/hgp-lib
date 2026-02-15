"""Core metrics dataclasses for generation-level metrics."""

from dataclasses import dataclass
from typing import Sequence

from ..rules import Rule


@dataclass()
class GenerationMetrics:
    """All metrics captured at a single generation for one population."""

    best_idx: int
    best_rule: Rule

    complexities: Sequence[int]
    train_scores: Sequence[float]
    child_population_generation_metrics: Sequence["GenerationMetrics"]

    val_score: float | None = None

    @classmethod
    def from_population(
        cls,
        best_idx: int,
        best_rule: Rule,
        train_scores: Sequence[float],
        complexities: Sequence[int],
        child_population_generation_metrics: Sequence["GenerationMetrics"],
    ) -> "GenerationMetrics":
        return cls(
            best_rule=best_rule,
            best_idx=best_idx,
            complexities=complexities,
            train_scores=train_scores,
            child_population_generation_metrics=(child_population_generation_metrics),
        )

    @property
    def best_train_score(self) -> float:
        return self.train_scores[self.best_idx]

    @property
    def best_rule_complexity(self) -> int:
        return self.complexities[self.best_idx]

    @property
    def population_size(self) -> int:
        return len(self.train_scores)
