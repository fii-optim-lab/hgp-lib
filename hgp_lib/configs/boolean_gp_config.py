from dataclasses import dataclass
from typing import Callable

from numpy import ndarray

from ..crossover import CrossoverExecutor
from ..mutations import MutationExecutor
from ..populations import PopulationGenerator
from ..rules import Rule
from ..selections import BaseSelection
from ..utils.validation import check_isinstance, check_X_y, validate_callable


@dataclass
class BooleanGPConfig:
    """
    Configuration for BooleanGP.

    Attributes:
        train_data (ndarray): Training data (2D boolean array).
        train_labels (ndarray): Training labels (1D integer array).
        score_fn (Callable): Fitness function (predictions, labels) -> float.
        population_generator (PopulationGenerator | None): Optional; default created from num_features.
        mutation_executor (MutationExecutor | None): Optional; default created from num_features.
        crossover_executor (CrossoverExecutor | None): Optional; default CrossoverExecutor().
        selection (BaseSelection | None): Optional; default RouletteSelection().
        optimize_scorer (bool): Whether to optimize scorer per data (dedupe + sample weights).
        regeneration (bool): Whether to regenerate population on plateau.
        regeneration_patience (int): Epochs without improvement before regeneration.
        check_valid (Callable[[Rule], bool] | None): Optional rule validator for mutation/crossover.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.configs import BooleanGPConfig
        >>> data = np.array([[True, False], [False, True], [True, True], [False, False]])
        >>> labels = np.array([1, 0, 1, 0])
        >>> def accuracy(p, l): return float((p == l).mean())
        >>> config = BooleanGPConfig(train_data=data, train_labels=labels, score_fn=accuracy)
        >>> config.train_data.shape
        (4, 2)
        >>> config.optimize_scorer
        False
    """

    train_data: ndarray
    train_labels: ndarray
    score_fn: Callable[[ndarray, ndarray], float]
    population_generator: PopulationGenerator | None = None
    mutation_executor: MutationExecutor | None = None
    crossover_executor: CrossoverExecutor | None = None
    selection: BaseSelection | None = None
    optimize_scorer: bool = False
    regeneration: bool = False
    regeneration_patience: int = 100
    check_valid: Callable[[Rule], bool] | None = None


def validate_gp_config(config: BooleanGPConfig) -> None:
    """
    Validate BooleanGPConfig.

    Args:
        config (BooleanGPConfig): Configuration to validate.

    Raises:
        TypeError: If any field has incorrect type.
        ValueError: If any field has invalid value.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.configs import BooleanGPConfig
        >>> from hgp_lib.configs.boolean_gp_config import validate_gp_config
        >>> data = np.array([[True, False], [False, True]])
        >>> labels = np.array([1, 0])
        >>> def accuracy(p, l): return float((p == l).mean())
        >>> config = BooleanGPConfig(train_data=data, train_labels=labels, score_fn=accuracy)
        >>> validate_gp_config(config)  # No error
    """
    check_X_y(config.train_data, config.train_labels)
    validate_callable(config.score_fn)
    check_isinstance(config.regeneration, bool)
    check_isinstance(config.regeneration_patience, int)
    if config.regeneration and config.regeneration_patience < 1:
        raise ValueError("regeneration_patience must be a positive integer")
    if config.population_generator is not None:
        check_isinstance(config.population_generator, PopulationGenerator)
    if config.mutation_executor is not None:
        check_isinstance(config.mutation_executor, MutationExecutor)
    if config.crossover_executor is not None:
        check_isinstance(config.crossover_executor, CrossoverExecutor)
    if config.selection is not None:
        check_isinstance(config.selection, BaseSelection)
    if config.check_valid is not None:
        validate_callable(config.check_valid)
