from typing import List, Sequence, Tuple

import numpy as np
from numpy import ndarray

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
            Default: `0.4`.

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
        >>> selected_rules, selected_scores = selection.select(rules, scores, 3)
        >>> len(selected_rules)
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
    ) -> Tuple[List[Rule], ndarray]:
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
            Tuple[List[Rule], ndarray]: A tuple containing:
                - List[Rule]: Copies of the selected rules. The same rule may appear multiple times.
                - ndarray: The fitness scores of the selected rules.

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
            >>> selected_rules, selected_scores = selection.select(rules, scores, 2)
            >>> len(selected_rules)
            2
            >>> all(isinstance(rule, Rule) for rule in selected_rules)
            True
        """
        n = len(rules)

        scores_array = np.asarray(scores)
        sorted_order = np.argsort(-scores_array)

        winning_seats = np.random.choice(
            self.tournament_size, size=n_select, p=self.probs
        )

        # Vectorized tournament sampling: for each of n_select tournaments,
        # pick tournament_size unique indices from [0, n) by selecting the
        # tournament_size smallest random keys per row.
        random_keys = np.random.random((n_select, n))
        tournament_matrix = np.argpartition(
            random_keys, self.tournament_size - 1, axis=1
        )[:, : self.tournament_size]
        tournament_matrix.sort(axis=1)

        # Select the winning rank from each tournament, then map to the
        # original population index via sorted_order.
        winner_ranks = tournament_matrix[np.arange(n_select), winning_seats]
        winner_indices = sorted_order[winner_ranks]

        selected_rules = [rules[idx].copy() for idx in winner_indices]
        selected_scores = scores_array[winner_indices]
        return selected_rules, selected_scores
