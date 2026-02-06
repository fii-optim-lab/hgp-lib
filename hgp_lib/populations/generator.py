from typing import Sequence, List

import numpy as np
from numpy.random import Generator

from .base_strategy import PopulationStrategy
from ..rules import Rule
from ..utils.validation import check_isinstance


class PopulationGenerator:
    """
    Generates a population of rules using one or more strategies with weighted probability.

    Attributes:
        strategies (Sequence[PopulationStrategy]): The list of strategies to use.
        population_size (int): The total number of rules to generate. Default: `100`.
        weights (Sequence[float] | np.ndarray| None): Weights for random selection of strategies.
            If `None`, all strategies are selected with equal probability. Default: `None`.

    Examples:
        >>> from numpy.random import default_rng
        >>> from hgp_lib.populations import PopulationGenerator, RandomStrategy
        >>> rng = default_rng(42)
        >>> strategy = RandomStrategy(num_literals=5)
        >>> generator = PopulationGenerator(strategies=[strategy], population_size=10)
        >>> population = generator.generate(rng)
        >>> len(population)
        10
    """

    def __init__(
        self,
        strategies: Sequence[PopulationStrategy],
        population_size: int = 100,
        weights: Sequence[float] | np.ndarray | None = None,
    ):
        """
        Initialize the PopulationGenerator.

        Args:
            strategies (Sequence[PopulationStrategy]): A non-empty sequence of PopulationStrategy instances.
            population_size (int): The number of rules to generate. Must be greater than `0`. Default: `100`.
            weights (Sequence[float] | np.ndarray | None): Optional weights for each strategy.
                Must sum to > `0` and be non-negative. Default: `None`.
        """
        check_isinstance(population_size, int)
        check_isinstance(strategies, Sequence)

        if len(strategies) == 0:
            raise ValueError("Strategies must be a non-empty Sequence")
        for strategy in strategies:
            check_isinstance(strategy, PopulationStrategy)

        if population_size <= 0:
            raise ValueError(
                f"population_size must be a positive integer, got {population_size}"
            )

        self.strategies = strategies
        self.population_size = population_size

        self._pvals = self._init_pvals(weights)

    def _init_pvals(self, weights: Sequence[float] | np.ndarray | None) -> np.ndarray:
        if weights is not None:
            if isinstance(weights, np.ndarray):
                weights = weights.tolist()
            check_isinstance(weights, Sequence)
            if len(weights) != len(self.strategies):
                raise ValueError(
                    f"weights length ({len(weights)}) must match strategies length ({len(self.strategies)})"
                )
            if any(w < 0 for w in weights):
                raise ValueError("weights must be non-negative")
            if sum(weights) <= 0:
                raise ValueError("Sum of weights must be positive")

        pvals = (
            weights
            if weights is not None
            else [1.0 / len(self.strategies)] * len(self.strategies)
        )
        sum_weights = sum(pvals)
        return np.array([w / sum_weights for w in pvals])

    def generate(self, rng: Generator) -> List[Rule]:
        """
        Generates the full population of rules.

        Args:
            rng (Generator): NumPy random Generator for reproducible randomness.

        Returns:
            List[Rule]: A list containing `population_size` generated rules.
        """
        counts = rng.multinomial(self.population_size, self._pvals)
        population = []
        for strategy, count in zip(self.strategies, counts):
            if count > 0:
                population.extend(strategy.generate(count, rng))
        return population
