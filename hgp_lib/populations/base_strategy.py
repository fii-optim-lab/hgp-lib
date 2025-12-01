from abc import ABC, abstractmethod
from typing import List
from ..rules import Rule


class PopulationStrategy(ABC):
    """
    Abstract base class for population generation strategies.

    Strategies define how individual rules are created for the initial population.
    Concrete implementations must define the `generate` method.
    """

    @abstractmethod
    def generate(self, n: int) -> List[Rule]:
        """
        Generates n rules according to the strategy.

        Args:
            n (int): Number of rules to generate.

        Returns:
            List[Rule]: A list of newly generated rule instances.
        """
        pass
