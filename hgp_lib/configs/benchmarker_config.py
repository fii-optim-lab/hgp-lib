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
class BenchmarkerConfig:
    """
    Configuration for GPBenchmarker. Used for multi-run benchmarking with k-fold CV.

    Attributes:
        data (ndarray): Full dataset (split internally).
        labels (ndarray): Labels.
        score_fn (Callable): Fitness function.
        num_epochs (int): Epochs per fold.
        population_generator (PopulationGenerator | None): Optional.
        mutation_executor (MutationExecutor | None): Optional.
        crossover_executor (CrossoverExecutor | None): Optional.
        selection (BaseSelection | None): Optional.
        optimize_scorer (bool): Optimize scorer per fold.
        regeneration (bool): Regeneration on plateau.
        regeneration_patience (int): Patience before regeneration.
        val_score_fn (Callable | None): Validation scorer.
        check_valid (Callable | None): Rule validator for mutation/crossover.
        num_runs (int): Number of benchmark runs.
        test_size (float): Fraction for test set.
        n_folds (int): K for k-fold CV.
        n_jobs (int): Parallel jobs (-1 = all CPUs).
        base_seed (int): Base random seed.
        val_every (int): Validation frequency in epochs.
        progress_bar (bool): Master progress switch.
        show_run_progress (bool): Progress for runs.
        show_fold_progress (bool): Progress for folds.
        show_epoch_progress (bool): Progress for epochs.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.configs import BenchmarkerConfig
        >>> data = np.array([[True, False], [False, True], [True, True], [False, False]])
        >>> labels = np.array([1, 0, 1, 0])
        >>> def accuracy(p, l): return float((p == l).mean())
        >>> config = BenchmarkerConfig(data=data, labels=labels, score_fn=accuracy, num_epochs=10)
        >>> config.num_runs
        30
        >>> config.n_folds
        5
    """

    data: ndarray
    labels: ndarray
    score_fn: Callable[[ndarray, ndarray], float]
    num_epochs: int
    population_generator: PopulationGenerator | None = None
    mutation_executor: MutationExecutor | None = None
    crossover_executor: CrossoverExecutor | None = None
    selection: BaseSelection | None = None
    optimize_scorer: bool = True
    regeneration: bool = False
    regeneration_patience: int = 100
    val_score_fn: Callable[[ndarray, ndarray], float] | None = None
    check_valid: Callable[[Rule], bool] | None = None
    num_runs: int = 30
    test_size: float = 0.2
    n_folds: int = 5
    n_jobs: int = -1
    base_seed: int = 0
    val_every: int = 100
    progress_bar: bool = True
    show_run_progress: bool = True
    show_fold_progress: bool = True
    show_epoch_progress: bool = True


def validate_benchmarker_config(config: BenchmarkerConfig) -> None:
    """
    Validate BenchmarkerConfig.

    Args:
        config (BenchmarkerConfig): Configuration to validate.

    Raises:
        TypeError: If any field has incorrect type.
        ValueError: If any field has invalid value.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.configs import BenchmarkerConfig
        >>> from hgp_lib.configs.benchmarker_config import validate_benchmarker_config
        >>> data = np.array([[True, False], [False, True], [True, True], [False, False]])
        >>> labels = np.array([1, 0, 1, 0])
        >>> def accuracy(p, l): return float((p == l).mean())
        >>> config = BenchmarkerConfig(data=data, labels=labels, score_fn=accuracy, num_epochs=10)
        >>> validate_benchmarker_config(config)  # No error
    """
    check_X_y(config.data, config.labels)
    validate_callable(config.score_fn)
    check_isinstance(config.num_epochs, int)
    if config.num_epochs < 1:
        raise ValueError("num_epochs must be a positive integer")
    check_isinstance(config.num_runs, int)
    if config.num_runs < 1:
        raise ValueError("num_runs must be a positive integer")
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    check_isinstance(config.n_folds, int)
    if config.n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if config.check_valid is not None:
        validate_callable(config.check_valid)
    if config.population_generator is not None:
        check_isinstance(config.population_generator, PopulationGenerator)
    if config.mutation_executor is not None:
        check_isinstance(config.mutation_executor, MutationExecutor)
    if config.crossover_executor is not None:
        check_isinstance(config.crossover_executor, CrossoverExecutor)
    if config.selection is not None:
        check_isinstance(config.selection, BaseSelection)
