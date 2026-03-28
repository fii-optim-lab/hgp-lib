"""History dataclasses for tracking population evolution over time."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import List


from ..rules import Rule
from .core import GenerationMetrics


@dataclass
class PopulationHistory:
    """Complete history of a population across all generations."""

    global_best_rule: Rule
    tp: int
    fp: int
    fn: int
    tn: int
    val_tp: int | None = None
    val_fp: int | None = None
    val_fn: int | None = None
    val_tn: int | None = None
    generations: List[GenerationMetrics] = field(default_factory=list)

    @property
    def __len__(self) -> int:
        return len(self.generations)

    @cached_property
    def best_val_score(self):
        val_scores = [x.val_score for x in self.generations if x.val_score is not None]
        if len(val_scores) == 0:
            return None  # Can't be None if validation data exists.
        return max(val_scores)
