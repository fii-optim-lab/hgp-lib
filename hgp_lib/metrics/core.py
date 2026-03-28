"""Core metrics dataclasses for generation-level metrics."""

from dataclasses import dataclass
from typing import Sequence

from ..rules import Rule


@dataclass()
class GenerationMetrics:
    """
    Metrics captured at a single generation for one population.

    Stores per-rule training scores, complexities, the best rule found in this
    generation, and optionally a validation score. In hierarchical GP, child
    population metrics are nested via ``child_population_generation_metrics``.

    Args:
        best_idx (int):
            Index of the best-scoring rule in ``train_scores``.
        best_rule (Rule):
            Copy of the best rule from this generation.
        complexities (Sequence[int]):
            Number of nodes in each rule (same order as ``train_scores``).
        train_scores (Sequence[float]):
            Fitness scores for every rule in the population.
        child_population_generation_metrics (Sequence[GenerationMetrics]):
            Metrics from child populations in hierarchical GP. Empty list for
            flat (non-hierarchical) runs.
        val_score (float | None):
            Validation score of the global best rule at this generation, or ``None``
            if validation was not performed. Default: `None`.

    Examples:
        >>> from hgp_lib.metrics import GenerationMetrics
        >>> from hgp_lib.rules import Literal
        >>> m = GenerationMetrics.from_population(
        ...     best_idx=1,
        ...     best_rule=Literal(value=1),
        ...     train_scores=[0.7, 0.9, 0.5],
        ...     complexities=[1, 3, 2],
        ...     child_population_generation_metrics=[],
        ... )
        >>> m.best_train_score
        0.9
        >>> m.best_rule_complexity
        3
        >>> m.population_size
        3
    """

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
        """
        Construct a ``GenerationMetrics`` from population-level data.

        This is the preferred constructor used by ``BooleanGP._new_generation``.

        Args:
            best_idx (int):
                Index of the best rule in ``train_scores``.
            best_rule (Rule):
                The best rule (already copied).
            train_scores (Sequence[float]):
                Per-rule fitness scores.
            complexities (Sequence[int]):
                Per-rule node counts.
            child_population_generation_metrics (Sequence[GenerationMetrics]):
                Child metrics (empty list for flat GP).

        Returns:
            GenerationMetrics: A new instance with ``val_score=None``.

        Examples:
            >>> from hgp_lib.metrics import GenerationMetrics
            >>> from hgp_lib.rules import Literal
            >>> m = GenerationMetrics.from_population(
            ...     best_idx=0, best_rule=Literal(value=0),
            ...     train_scores=[0.8], complexities=[1],
            ...     child_population_generation_metrics=[],
            ... )
            >>> m.val_score is None
            True
        """
        return cls(
            best_rule=best_rule,
            best_idx=best_idx,
            complexities=complexities,
            train_scores=train_scores,
            child_population_generation_metrics=child_population_generation_metrics,
        )

    @property
    def best_train_score(self) -> float:
        """
        Training score of the best rule in this generation.

        Examples:
            >>> from hgp_lib.metrics import GenerationMetrics
            >>> from hgp_lib.rules import Literal
            >>> m = GenerationMetrics.from_population(
            ...     best_idx=2, best_rule=Literal(value=0),
            ...     train_scores=[0.1, 0.2, 0.9], complexities=[1, 1, 1],
            ...     child_population_generation_metrics=[],
            ... )
            >>> m.best_train_score
            0.9
        """
        return self.train_scores[self.best_idx]

    @property
    def best_rule_complexity(self) -> int:
        """
        Node count of the best rule in this generation.

        Examples:
            >>> from hgp_lib.metrics import GenerationMetrics
            >>> from hgp_lib.rules import Literal
            >>> m = GenerationMetrics.from_population(
            ...     best_idx=0, best_rule=Literal(value=0),
            ...     train_scores=[0.8], complexities=[5],
            ...     child_population_generation_metrics=[],
            ... )
            >>> m.best_rule_complexity
            5
        """
        return self.complexities[self.best_idx]

    @property
    def population_size(self) -> int:
        """
        Number of rules in the population at this generation.

        Examples:
            >>> from hgp_lib.metrics import GenerationMetrics
            >>> from hgp_lib.rules import Literal
            >>> m = GenerationMetrics.from_population(
            ...     best_idx=0, best_rule=Literal(value=0),
            ...     train_scores=[0.1, 0.2, 0.3], complexities=[1, 2, 3],
            ...     child_population_generation_metrics=[],
            ... )
            >>> m.population_size
            3
        """
        return len(self.train_scores)
