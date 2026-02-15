"""Core metrics dataclasses for generation-level metrics."""

from dataclasses import dataclass, field
from typing import List, Tuple

from ..rules import Rule


@dataclass(frozen=True)
class GenerationMetrics:
    """All metrics captured at a single generation for one population."""

    generation: int
    population_size: int
    best_idx: int
    best_rule: Rule

    complexities: Tuple[int, ...]
    train_scores: Tuple[float, ...]

    val_score: float | None = None

    # Hierarchical GP
    child_population_generation_metrics: Tuple["GenerationMetrics", ...] = field(default_factory=tuple)

    @classmethod
    def from_population(
        cls,
        generation: int,
        best_idx: int,
        best_rule: Rule,
        train_scores: List[float],
        complexities: List[int],
        child_population_generation_metrics: List["GenerationMetrics"],
    ) -> "GenerationMetrics":

        return cls(
            generation=generation,
            best_rule=best_rule,
            best_idx=best_idx,
            population_size=len(train_scores),
            complexities=tuple(complexities),
            train_scores=tuple(train_scores),
            child_population_generation_metrics=(
                tuple(child_population_generation_metrics)
            ),
        )

    @property
    def best_train_score(self) -> Rule:
        return self.train_scores[self.best_idx]

    @property
    def best_rule_complexity(self) -> int:
        return self.complexities[self.best_idx]
