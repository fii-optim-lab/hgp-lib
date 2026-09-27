from ..evaluation import Evaluator
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

        Subclass to use custom strategies. The evaluator holds the population's training
        rows and its scorer:

        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset, NumpyBackend, resolve_scorer
        >>> from hgp_lib.populations import PopulationGeneratorFactory, BestLiteralStrategy
        >>> class MyFactory(PopulationGeneratorFactory):
        ...     def create_strategies(self, num_literals, evaluator):
        ...         return [BestLiteralStrategy(num_literals=num_literals, evaluator=evaluator)]
        >>> factory = MyFactory(population_size=20)
        >>> data = np.array([[True, False], [False, True]])
        >>> evaluator = NumpyBackend().bind(Dataset(data, np.array([1, 0])), resolve_scorer())
        >>> gen = factory.create(2, evaluator)
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
        self, num_literals: int, evaluator: Evaluator
    ) -> list[PopulationStrategy]:
        """
        Create the list of strategies for the generator.

        Override this method to use custom strategies. The default creates
        a single `RandomStrategy(num_literals=num_literals)`.

        Args:
            num_literals (int): Number of boolean features (columns of the training data).
            evaluator (Evaluator): The population's training rows (``evaluator.dataset``,
                possibly merged into sample weights) with its scorer bound to them. Use
                ``evaluator.score`` to score rules on all rows, and ``evaluator.scorer``
                on subsets from ``evaluator.dataset.take``.

        Returns:
            list[PopulationStrategy]: Strategies to pass to `PopulationGenerator`.
        """
        return [RandomStrategy(num_literals=num_literals)]

    def create(self, num_literals: int, evaluator: Evaluator) -> PopulationGenerator:
        """
        Create a `PopulationGenerator` with data-dependent strategies.

        Args:
            num_literals (int): Number of boolean features (columns of the training data).
            evaluator (Evaluator): The population's training rows with its scorer.

        Returns:
            PopulationGenerator: A generator ready to produce the initial population.
        """
        strategies = self.create_strategies(num_literals, evaluator)
        return PopulationGenerator(
            strategies=strategies, population_size=self.population_size
        )
