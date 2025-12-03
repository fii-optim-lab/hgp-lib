from typing import Tuple

from ..rules import Rule


class StubCrossover:
    """
    Trivial crossover operator that simply copies the two parents.

    The class mirrors the shape of a real crossover executor so it can be
    swapped in tests or minimal setups where no crossover is desired.
    """

    def crossover(self, parent_a: Rule, parent_b: Rule) -> Tuple[Rule, Rule]:
        """
        Returns deep copies of the received parents.

        Args:
            parent_a (Rule): First parent.
            parent_b (Rule): Second parent.

        Returns:
            Tuple[Rule, Rule]: Independent copies of the parents.
        """
        return parent_a.copy(), parent_b.copy()

