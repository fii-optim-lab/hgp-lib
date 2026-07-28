from collections.abc import Callable

import numpy as np

from ..utils.validation import check_isinstance
from .base_strategy import PopulationStrategy
from .generator import PopulationGenerator
from .strategies import RandomStrategy


class PopulationGeneratorFactory:
    """
    Factory for creating `PopulationGenerator` instances.

    Stores configuration-time parameters (`population_size`) and defers
    data-dependent construction to `create`. Override `create_strategies`
    to customise which strategies are instantiated.

    Attributes:
        population_size (int): Number of rules the generator will produce.
            Default: `100`.

    Examples:
        >>> from hgp_lib.populations import PopulationGeneratorFactory
        >>> factory = PopulationGeneratorFactory(population_size=50)
        >>> factory.population_size
        50

        Subclass to use custom strategies:

        >>> from sklearn.metrics import accuracy_score
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
        >>> gen = factory.create(2, accuracy_score, data, labels)
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
    ) -> list[PopulationStrategy]:
        """
        Create the list of strategies for the generator.

        Override this method to use custom strategies. The default creates
        a single `RandomStrategy(num_literals=num_literals)`.

        Args:
            num_literals (int): Number of boolean features (columns in train_data).
            score_fn (Callable): Fitness function `(y_true, y_pred) -> float`.
            train_data (np.ndarray): Training data (2-D boolean array).
            train_labels (np.ndarray): Training labels (1-D array).

        Returns:
            list[PopulationStrategy]: Strategies to pass to `PopulationGenerator`.
        """
        return [RandomStrategy(num_literals=num_literals)]

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
            score_fn (Callable): Fitness function `(y_true, y_pred) -> float`.
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
