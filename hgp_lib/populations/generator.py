from typing import Sequence, List, Optional

import numpy as np

from .base_strategy import PopulationStrategy
from ..rules import Rule


class PopulationGenerator:
    """
    Generates a population of rules using one or more strategies with weighted probability.

    Attributes:
        strategies (Sequence[PopulationStrategy]): The list of strategies to use.
        population_size (int): The total number of rules to generate. Default: `100`.
        weights (Sequence[float] | None): Weights for random selection of strategies.
            If `None`, all strategies are selected with equal probability. Default: `None`.

    Examples:
        >>> from hgp_lib.populations import PopulationGenerator, RandomStrategy
        >>> strategy = RandomStrategy(num_literals=5)
        >>> generator = PopulationGenerator(strategies=[strategy], population_size=10)
        >>> population = generator.generate()
        >>> len(population)
        10
    """

    def __init__(
        self,
        strategies: Sequence[PopulationStrategy],
        population_size: int = 100,
        weights: Optional[Sequence[float]] = None,
    ):
        """
        Initialize the PopulationGenerator.

        Args:
            strategies (Sequence[PopulationStrategy]): A non-empty sequence of PopulationStrategy instances.
            population_size (int): The number of rules to generate. Must be greater than 0. Default: `100`.
            weights (Optional[Sequence[float]]): Optional weights for each strategy. Must sum to > 0 and be non-negative.
                Default: `None`.
        """
        if not isinstance(strategies, Sequence) or len(strategies) == 0:
            raise ValueError("Strategies must be a non-empty Sequence")
        for s in strategies:
            if not isinstance(s, PopulationStrategy):
                raise TypeError(
                    f"All elements in strategies must be PopulationStrategy, found {type(s)}"
                )

        if not isinstance(population_size, int) or population_size <= 0:
            raise ValueError(
                f"population_size must be a positive integer, got {population_size}"
            )

        if weights is not None:
            if not isinstance(weights, Sequence):
                raise TypeError(f"weights must be a Sequence, got {type(weights)}")
            if len(weights) != len(strategies):
                raise ValueError(
                    f"weights length ({len(weights)}) must match strategies length ({len(strategies)})"
                )
            if any(w < 0 for w in weights):
                raise ValueError("weights must be non-negative")
            if sum(weights) <= 0:
                raise ValueError("Sum of weights must be positive")

        self.strategies = strategies
        self.population_size = population_size

        pvals = (
            weights
            if weights is not None
            else [1.0 / len(strategies)] * len(strategies)
        )
        sum_weights = sum(pvals)
        pvals = [w / sum_weights for w in pvals]
        self.counts = np.random.multinomial(self.population_size, pvals)

    def generate(self) -> List[Rule]:
        """
        Generates the full population of rules.

        Returns:
            List[Rule]: A list containing `population_size` generated rules.
        """
        population = []
        for strategy, count in zip(self.strategies, self.counts):
            if count > 0:
                population.extend(strategy.generate(count))
        return population
