from dataclasses import dataclass
from typing import Callable

from numpy import ndarray

from ..crossover import CrossoverExecutor
from ..mutations import MutationExecutor
from ..populations import PopulationGenerator, SamplingStrategy
from ..rules import Rule
from ..selections import BaseSelection
from ..utils.validation import check_isinstance, check_X_y, validate_callable


@dataclass
class BooleanGPConfig:
    """
    Configuration for BooleanGP.

    Attributes:
        score_fn (Callable): Fitness function (predictions, labels) -> float.
        train_data (ndarray | None): Training data (2D boolean array). Can be None when
            used as a template in BenchmarkerConfig (data provided at benchmarker level).
        train_labels (ndarray | None): Training labels (1D integer array). Can be None
            when used as a template in BenchmarkerConfig.
        population_generator (PopulationGenerator | None): Optional; default created from num_features.
        mutation_executor (MutationExecutor | None): Optional; default created from num_features.
        crossover_executor (CrossoverExecutor | None): Optional; default CrossoverExecutor().
        selection (BaseSelection | None): Optional; default TournamentSelection().
        optimize_scorer (bool): Whether to optimize scorer via data deduplication and sample weights.
        regeneration (bool): Whether to regenerate population on plateau.
        regeneration_patience (int): Epochs without improvement before regeneration.
        check_valid (Callable[[Rule], bool] | None): Optional rule validator for mutation/crossover.
        num_child_populations (int): Number of child populations for hierarchical GP. Default: 0.
        max_depth (int): Maximum hierarchical depth; 0 means no children. Default: 0.
            Root population has current_depth=0, its children have current_depth=1, etc.
        sampling_strategy (SamplingStrategy | None): Strategy for sampling data/features for children.
            Required when max_depth > 0. Default: None.
        top_k_transfer (int): Number of top rules to transfer from each child to parent. Default: 10.
        feedback_type (str): How to apply parent feedback: "additive" or "multiplicative". Default: "multiplicative".
        feedback_strength (float): Coefficient for feedback signal. Must be > 0. Default: 0.1.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.configs import BooleanGPConfig
        >>> data = np.array([[True, False], [False, True], [True, True], [False, False]])
        >>> labels = np.array([1, 0, 1, 0])
        >>> def accuracy(p, l): return float((p == l).mean())
        >>> config = BooleanGPConfig(score_fn=accuracy, train_data=data, train_labels=labels)
        >>> config.train_data.shape
        (4, 2)
        >>> config.optimize_scorer
        True
    """

    # TODO: We should reconsider the ordering of the arguments for score fn. Pred, GT or GT, Pred?
    score_fn: Callable[[ndarray, ndarray], float]
    train_data: ndarray | None = None
    train_labels: ndarray | None = None
    population_generator: PopulationGenerator | None = None
    mutation_executor: MutationExecutor | None = None
    crossover_executor: CrossoverExecutor | None = None
    selection: BaseSelection | None = None
    optimize_scorer: bool = True
    regeneration: bool = False
    regeneration_patience: int = 100
    check_valid: Callable[[Rule], bool] | None = None
    # Hierarchical GP
    num_child_populations: int = 0
    max_depth: int = 0
    sampling_strategy: SamplingStrategy | None = None
    top_k_transfer: int = 10
    feedback_type: str = "multiplicative"
    feedback_strength: float = 0.1


def validate_gp_config(config: BooleanGPConfig, require_data: bool = True) -> None:
    """
    Validate BooleanGPConfig.

    Args:
        config (BooleanGPConfig): Configuration to validate.
        require_data (bool): If True, validates that train_data and train_labels are
            provided. Set to False when validating a template config (e.g., inside
            BenchmarkerConfig where data is provided separately). Default: `True`.

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
        >>> config = BooleanGPConfig(score_fn=accuracy, train_data=data, train_labels=labels)
        >>> validate_gp_config(config)  # No error
    """
    if require_data:
        if config.train_data is None or config.train_labels is None:
            raise ValueError("train_data and train_labels are required")
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

    # Hierarchical GP validation
    check_isinstance(config.num_child_populations, int)
    check_isinstance(config.max_depth, int)
    if config.num_child_populations < 0:
        raise ValueError("num_child_populations must be non-negative")
    if config.max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if config.max_depth > 0 and config.sampling_strategy is None:
        raise ValueError("sampling_strategy is required when max_depth > 0")
    if config.max_depth > 0 and config.num_child_populations == 0:
        raise ValueError("max_depth > 0 requires num_child_populations > 0")
    if config.sampling_strategy is not None:
        check_isinstance(config.sampling_strategy, SamplingStrategy)
    check_isinstance(config.top_k_transfer, int)
    if config.top_k_transfer < 1:
        raise ValueError("top_k_transfer must be at least 1")
    check_isinstance(config.feedback_type, str)
    if config.feedback_type not in ("additive", "multiplicative"):
        raise ValueError(
            f"feedback_type must be 'additive' or 'multiplicative', got {config.feedback_type!r}"
        )
    check_isinstance(config.feedback_strength, (int, float))
    if config.feedback_strength <= 0:
        raise ValueError("feedback_strength must be > 0")
