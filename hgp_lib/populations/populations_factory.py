from typing import Callable, List

import numpy as np

from .generator import PopulationGenerator
from .strategies import RandomStrategy
from .base_strategy import PopulationStrategy
from ..utils.validation import check_isinstance


class PopulationGeneratorFactory:
    """
    Factory for creating `PopulationGenerator` instances, supporting both binary and multiclass rule generation.

    Stores configuration-time parameters (`population_size`) and defers
    data-dependent construction to `create`. Override `create_strategies`
    to customise which strategies are instantiated. For multiclass, possible_classes are inferred from train_labels.

    Attributes:
        population_size (int): Number of rules the generator will produce.
            Default: `100`.

    Examples:
        >>> from hgp_lib.populations import PopulationGeneratorFactory
        >>> factory = PopulationGeneratorFactory(population_size=50)
        >>> factory.population_size
        50

        Subclass to use custom strategies:

        >>> import numpy as np
        >>> from hgp_lib.populations import PopulationGeneratorFactory, BestLiteralStrategy
        >>> class MyFactory(PopulationGeneratorFactory):
        ...     def create_strategies(self, num_literals, score_fn, train_data, train_labels):
        ...         return [BestLiteralStrategy(
        ...             num_literals=num_literals, score_fn=score_fn,
        ...             train_data=train_data, train_labels=train_labels,
        ...         )]
        >>> factory = MyFactory(population_size=20)
        >>> data = np.array([[True, False], [False, True]])
        >>> labels = np.array([1, 0])
        >>> def acc(p, l): return float((p == l).mean())
        >>> gen = factory.create(2, acc, data, labels)
        >>> len(gen.generate())
        20
    """

    def __init__(self, population_size: int = 100):
        check_isinstance(population_size, int)
        if population_size <= 0:
            raise ValueError(
                f"population_size must be a positive integer, got {population_size}"
            )
        self.population_size = population_size

    def create_strategies(
        self,
        num_literals: int,
        score_fn: Callable[[np.ndarray, np.ndarray], float],
        train_data: np.ndarray,
        train_labels: np.ndarray,
    ) -> List[PopulationStrategy]:
        """
        Create the list of strategies for the generator.

        Override this method to use custom strategies. The default creates
        a single `RandomStrategy(num_literals=num_literals)`.

        Args:
            num_literals (int): Number of boolean features (columns in train_data).
            score_fn (Callable): Fitness function `(predictions, labels) -> float`.
            train_data (np.ndarray): Training data (2-D boolean array).
            train_labels (np.ndarray): Training labels (1-D array).

        Returns:
            List[PopulationStrategy]: Strategies to pass to `PopulationGenerator`.
        """
        possible_classes = np.unique(train_labels)
        return [
            RandomStrategy(num_literals=num_literals, possible_classes=possible_classes)
        ]

    def create(
        self,
        num_literals: int,
        score_fn: Callable[[np.ndarray, np.ndarray], float],
        train_data: np.ndarray,
        train_labels: np.ndarray,
    ) -> PopulationGenerator:
        """
        Create a `PopulationGenerator` with data-dependent strategies.

        Args:
            num_literals (int): Number of boolean features (columns in train_data).
            score_fn (Callable): Fitness function `(predictions, labels) -> float`.
            train_data (np.ndarray): Training data (2-D boolean array).
            train_labels (np.ndarray): Training labels (1-D array).

        Returns:
            PopulationGenerator: A generator ready to produce the initial population.
        """
        strategies = self.create_strategies(
            num_literals, score_fn, train_data, train_labels
        )
        return PopulationGenerator(
            strategies=strategies, population_size=self.population_size
        )
