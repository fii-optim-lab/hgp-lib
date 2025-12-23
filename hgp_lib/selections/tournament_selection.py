from typing import List, Sequence

import numpy as np

from .base_selection import BaseSelection
from ..rules import Rule
from ..utils.validation import check_isinstance


class TournamentSelection(BaseSelection):
    """
    Selection strategy that holds tournaments among randomly sampled subsets of the population.

    The population is effectively sorted by fitness. For each selection event,
    `tournament_size` candidates are randomly sampled from the population.
    The candidate with the best rank in the sample is selected with probability `p`.
    If not selected, the second best is considered with probability `p`, and so on.

    The selection probability for the `i`-th ranked individual in a tournament is:
        `P(i) = selection_p * (1 - selection_p)^i`

    This creates selection pressure that favors fitter individuals while maintaining
    diversity. Higher `tournament_size` or `selection_p` increases selection pressure.

    Args:
        tournament_size (int):
            The number of candidates to include in each tournament. Must be greater
            than `0`. Default: `10`.
        selection_p (float):
            The probability of selecting the best candidate in the tournament.
            Must be between `0.0` and `1.0`. Lower values reduce selection pressure.
            Defaults to `0.4`.

    Raises:
        TypeError: If `tournament_size` is not an int or `selection_p` is not a float.
        ValueError: If `selection_p` is not in [0, 1] or `tournament_size` <= 0.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.selections import TournamentSelection
        >>> from hgp_lib.rules import Literal
        >>> np.random.seed(42)
        >>> selection = TournamentSelection(tournament_size=3, selection_p=0.5)
        >>> rules = [Literal(value=i) for i in range(5)]
        >>> scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        >>> selected = selection.select(rules, scores, 3)
        >>> len(selected)
        3
    """

    def __init__(self, tournament_size: int = 10, selection_p: float = 0.4):
        check_isinstance(tournament_size, int)
        check_isinstance(selection_p, float)
        if selection_p < 0 or selection_p > 1:
            raise ValueError(
                f"selection_p must be a float between 0 and 1, is {selection_p}"
            )
        if tournament_size <= 0:
            raise ValueError(
                f"tournament_size must be greater than 0, is {tournament_size}"
            )

        self.tournament_size = tournament_size
        self.selection_p = selection_p

        ranks = np.arange(tournament_size)
        probs = selection_p * ((1 - selection_p) ** ranks)
        # Normalize the tail: ensure the sum is exactly 1.0 by adding the remaining probability mass to the last element.
        probs[-1] += 1.0 - np.sum(probs)
        self.probs: np.ndarray = probs

    def select(
        self,
        rules: Sequence[Rule],
        scores: np.ndarray | Sequence[float],
        n_select: int,
    ) -> List[Rule]:
        """
        Selects `n_select` rules using tournament selection.

        The method performs `n_select` independent tournaments. In each tournament:
        1. `tournament_size` indices are sampled randomly from the population (without replacement).
        2. These indices are sorted by the fitness of the corresponding rules (best to worst).
        3. A winner is chosen based on the pre-calculated geometric probabilities.

        The caller must ensure that `len(rules) == `len(scores)` and `n_select >= self.tournament_size`.

        Args:
            rules (Sequence[Rule]):
                The collection of candidate rules to select from. Length must be
                greater than or equal to `tournament_size`.
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
            >>> from hgp_lib.selections import TournamentSelection
            >>> from hgp_lib.rules import Literal
            >>> random.seed(42); np.random.seed(42)
            >>> selection = TournamentSelection(tournament_size=3, selection_p=0.5)
            >>> rules = [
            ...     Literal(value=0),
            ...     Literal(value=1),
            ...     Literal(value=2),
            ... ]
            >>> scores = [0.2, 0.8, 0.5]
            >>> selected = selection.select(rules, scores, 2)
            >>> len(selected)
            2
            >>> all(isinstance(rule, Rule) for rule in selected)
            True
        """
        n = len(rules)

        scores = np.asarray(scores)
        sorted_order = np.argsort(-scores)

        # TODO: Measure this. Check if this can be precomputed.
        winning_seats = np.random.choice(
            self.tournament_size, size=n_select, p=self.probs
        )

        selected_rules = []

        for i in range(n_select):
            tournament_indices = np.random.choice(
                n, self.tournament_size, replace=False
            )
            tournament_indices.sort()
            selected_rules.append(
                rules[sorted_order[tournament_indices[winning_seats[i]]]].copy()
            )

        return selected_rules
