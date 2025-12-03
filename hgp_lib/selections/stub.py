from typing import List, Sequence

from ..rules import Rule


class StubSelection:
    """
    Deterministic selection strategy that keeps the highest scoring rules.

    The stub behaves like a fitness-proportionate selection where the `n_select`
    fittest individuals are copied to form the mating pool.
    """

    def select(
        self,
        rules: Sequence[Rule],
        scores: Sequence[float],
        n_select: int,
    ) -> List[Rule]:
        """
        Picks the top `n_select` rules sorted by their score in descending order.

        Args:
            rules (Sequence[Rule]): Candidate rules.
            scores (Sequence[float]): Scores corresponding to each rule.
            n_select (int): Number of rules to select. Must be between 1 and
                `len(rules)` (inclusive).

        Returns:
            List[Rule]: Copies of the best rules.
        """

        ranked = []
        for rule, score in zip(rules, scores):
            ranked.append((score, rule))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [rule.copy() for (_, rule) in ranked[:n_select]]
