from typing import Callable, List

import numpy as np

from .generator import PopulationGenerator
from .strategies import RandomStrategy
from .base_strategy import PopulationStrategy


def create_standard_population_strategies(
    num_literals: int,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    train_data: np.ndarray,
    test_data: np.ndarray,
) -> List[PopulationStrategy]:
    return [RandomStrategy(num_literals=num_literals)]


create_population_strategies_fn_type = Callable[
    [int, Callable[[np.ndarray, np.ndarray], float], np.ndarray, np.ndarray],
    List[PopulationStrategy],
]


def create_default_population_generator(
    population_size: int,
    num_literals: int,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    train_data: np.ndarray,
    test_data: np.ndarray,
) -> PopulationGenerator:
    strategies = create_standard_population_strategies(
        num_literals=num_literals,
        score_fn=score_fn,
        train_data=train_data,
        test_data=test_data,
    )
    return PopulationGenerator(strategies=strategies, population_size=population_size)


create_population_generator_fn_type = Callable[
    [int, int, Callable[[np.ndarray, np.ndarray], float], np.ndarray, np.ndarray],
    PopulationGenerator,
]
