"""History dataclasses for tracking population evolution over time."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import List


from ..rules import Rule
from .core import GenerationMetrics


@dataclass
class PopulationHistory:
    """
    Complete history of a population across all training generations.

    Stores the global best rule, training and validation confusion matrix values,
    and a list of ``GenerationMetrics`` — one per epoch. Used as the return type
    of ``GPTrainer.fit()`` and as fold-level results inside ``RunResult``.

    Args:
        global_best_rule (Rule):
            The best rule found across all generations (by validation score when
            available, otherwise by training score).
        tp (int): True positives of the global best rule on training data.
        fp (int): False positives of the global best rule on training data.
        fn (int): False negatives of the global best rule on training data.
        tn (int): True negatives of the global best rule on training data.
        val_tp (int | None): True positives on validation data, or ``None``. Default: `None`.
        val_fp (int | None): False positives on validation data, or ``None``. Default: `None`.
        val_fn (int | None): False negatives on validation data, or ``None``. Default: `None`.
        val_tn (int | None): True negatives on validation data, or ``None``. Default: `None`.
        generations (List[GenerationMetrics]):
            Per-epoch metrics. Default: empty list.

    Examples:
        >>> from hgp_lib.metrics import PopulationHistory, GenerationMetrics
        >>> from hgp_lib.rules import Literal
        >>> ph = PopulationHistory(
        ...     global_best_rule=Literal(value=0), tp=5, fp=1, fn=2, tn=7,
        ... )
        >>> len(ph.generations)
        0
        >>> ph.best_val_score is None
        True
    """

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
        """
        Maximum validation score across all generations, or ``None`` if no
        generation has a validation score.

        Examples:
            >>> from dataclasses import replace
            >>> from hgp_lib.metrics import PopulationHistory, GenerationMetrics
            >>> from hgp_lib.rules import Literal
            >>> g1 = GenerationMetrics.from_population(
            ...     best_idx=0, best_rule=Literal(value=0),
            ...     train_scores=[0.8], complexities=[1],
            ...     child_population_generation_metrics=[],
            ... )
            >>> g2 = replace(g1, val_score=0.6)
            >>> g3 = replace(g1, val_score=0.9)
            >>> ph = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
            ...     generations=[g1, g2, g3],
            ... )
            >>> ph.best_val_score
            0.9
        """
        val_scores = [x.val_score for x in self.generations if x.val_score is not None]
        if len(val_scores) == 0:
            return None
        return max(val_scores)

    @cached_property
    def best_train_score(self):
        """
        Maximum training score across all generations, or ``None`` if there are
        no generations.

        Examples:
            >>> from hgp_lib.metrics import PopulationHistory, GenerationMetrics
            >>> from hgp_lib.rules import Literal
            >>> g1 = GenerationMetrics.from_population(
            ...     best_idx=0, best_rule=Literal(value=0),
            ...     train_scores=[0.6], complexities=[1],
            ...     child_population_generation_metrics=[],
            ... )
            >>> g2 = GenerationMetrics.from_population(
            ...     best_idx=0, best_rule=Literal(value=0),
            ...     train_scores=[0.9], complexities=[1],
            ...     child_population_generation_metrics=[],
            ... )
            >>> ph = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
            ...     generations=[g1, g2],
            ... )
            >>> ph.best_train_score
            0.9
        """
        if len(self.generations) == 0:
            return None
        return max([g.best_train_score for g in self.generations])
