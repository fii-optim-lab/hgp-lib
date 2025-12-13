from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from ..rules import Rule


class BaseSelection(ABC):
    """
    Abstract base class for selection strategies in genetic programming.

    All selection implementations must inherit from this class and implement
    the `select` method to choose rules from a population based on their fitness scores.

    Examples:
        >>> from hgp_lib.selections import BaseSelection
        >>> from hgp_lib.rules import Literal
        >>> class TopSelection(BaseSelection):
        ...     def select(self, rules, scores, n_select):
        ...         ranked = sorted(zip(scores, rules), reverse=True)
        ...         return [rule.copy() for _, rule in ranked[:n_select]]
        ...
        >>> selection = TopSelection()
        >>> rules = [Literal(value=0), Literal(value=1)]
        >>> scores = [0.5, 0.8]
        >>> selected = selection.select(rules, scores, 1)
        >>> len(selected)
        1
        >>> selected
        [1]
    """

    @abstractmethod
    def select(
        self,
        rules: Sequence[Rule],
        scores: np.ndarray | Sequence[float],
        n_select: int,
    ) -> List[Rule]:
        """
        Selects `n_select` rules from the population based on their fitness scores.

        Higher scores indicate better fitness. The method should return copies of
        the selected rules to avoid modifying the original population.

        Args:
            rules (Sequence[Rule]):
                The collection of candidate rules to select from.
            scores (np.ndarray | Sequence[float]):
                Fitness scores corresponding to each rule. A higher score indicates
                better fitness. Must have the same length as `rules`.
            n_select (int):
                Number of rules to select. Must be between 1 and `len(rules)` (inclusive).

        Returns:
            List[Rule]: Copies of the selected rules.

        Examples:
            >>> from hgp_lib.selections import BaseSelection
            >>> from hgp_lib.rules import Literal
            >>> class TopSelection(BaseSelection):
            ...     def select(self, rules, scores, n_select):
            ...         ranked = sorted(zip(scores, rules), reverse=True)
            ...         return [rule.copy() for _, rule in ranked[:n_select]]
            ...
            >>> selection = TopSelection()
            >>> rules = [Literal(value=0), Literal(value=1), Literal(value=2)]
            >>> scores = [0.3, 0.9, 0.5]
            >>> selected = selection.select(rules, scores, 2)
            >>> len(selected)
            2
            >>> selected
            [1, 2]
        """
        pass
