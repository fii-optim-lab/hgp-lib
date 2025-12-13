from typing import List, Sequence

import numpy as np

from ..rules import Rule
from .base_selection import BaseSelection


class RouletteSelection(BaseSelection):
    """
    Fitness-proportionate selection using roulette wheel sampling.

    Each rule's probability of being selected is proportional to its fitness score.
    Higher scores receive proportionally higher selection probabilities. Negative scores
    are handled by shifting all scores to be non-negative before computing probabilities.
    Selection is performed with replacement, so the same rule can appear multiple times
    in the result.

    Examples:
        >>> import random
        >>> import numpy as np
        >>> from hgp_lib.selections import RouletteSelection
        >>> from hgp_lib.rules import Literal
        >>> random.seed(42); np.random.seed(42)
        >>> selection = RouletteSelection()
        >>> rules = [
        ...     Literal(value=0),
        ...     Literal(value=1),
        ...     Literal(value=2),
        ... ]
        >>> scores = [0.1, 0.5, 0.4]
        >>> selected = selection.select(rules, scores, 2)
        >>> len(selected)
        2
    """

    def select(
        self,
        rules: Sequence[Rule],
        scores: np.ndarray | Sequence[float],
        n_select: int,
    ) -> List[Rule]:
        """
        Selects `n_select` rules using roulette wheel (fitness-proportionate) selection.

        The probability of selecting each rule is proportional to its fitness score.
        If any scores are negative, all scores are shifted to be non-negative before
        computing probabilities. When all scores are equal, selection is uniform.
        Selection is performed with replacement, so the same rule may appear multiple times.

        Args:
            rules (Sequence[Rule]):
                The collection of candidate rules to select from.
            scores (np.ndarray | Sequence[float]):
                Fitness scores corresponding to each rule. Higher scores indicate better
                fitness. Must have the same length as `rules`.
            n_select (int):
                Number of rules to select.

        Returns:
            List[Rule]: Copies of the selected rules. The same rule may appear multiple times.

        Examples:
            >>> import random
            >>> import numpy as np
            >>> from hgp_lib.selections import RouletteSelection
            >>> from hgp_lib.rules import Literal
            >>> random.seed(42); np.random.seed(42)
            >>> selection = RouletteSelection()
            >>> rules = [
            ...     Literal(value=0),
            ...     Literal(value=1),
            ...     Literal(value=2),
            ... ]
            >>> scores = [0.1, 0.5, 0.4]
            >>> selected = selection.select(rules, scores, 2)
            >>> len(selected)
            2
            >>> all(isinstance(rule, Rule) for rule in selected)
            True
        """
        if len(rules) == 0:
            return []

        scores_array = np.asarray(scores)
        min_score = np.min(scores_array)
        if min_score < 0:
            scores_array = scores_array - min_score

        total = np.sum(scores_array)
        if total == 0 or np.all(scores_array == scores_array[0]):
            probabilities = np.ones(len(scores_array)) / len(scores_array)
        else:
            probabilities = scores_array / total

        selected_indices = np.random.choice(
            len(rules), size=n_select, p=probabilities, replace=True
        )

        return [rules[i] for i in selected_indices]
