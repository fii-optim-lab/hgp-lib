from dataclasses import dataclass

from numpy import ndarray

from ..utils.validation import check_isinstance, check_X_y
from .trainer_config import TrainerConfig, validate_trainer_config


@dataclass
class BenchmarkerConfig:
    """
    Configuration for GPBenchmarker. Used for multi-run benchmarking with k-fold CV.

    Contains a TrainerConfig template that specifies all training settings. The benchmarker
    will create TrainerConfig instances for each fold, replacing the data with fold-specific
    train/validation splits.

    Attributes:
        data (ndarray): Full dataset to be split into train/test and k-fold CV.
        labels (ndarray): Labels for the full dataset.
        trainer_config (TrainerConfig): Template configuration for training. The nested
            gp_config does not need train_data/train_labels (they will be set per fold).
        num_runs (int): Number of benchmark runs with different random seeds.
        test_size (float): Fraction of data to hold out for testing.
        n_folds (int): Number of folds for k-fold cross-validation.
        n_jobs (int): Number of parallel jobs (-1 = all CPUs, 1 = sequential).
        base_seed (int): Base random seed; each run uses base_seed + run_id.
        show_run_progress (bool): Show progress bar for runs.
        show_fold_progress (bool): Show progress bar for folds within each run.
        show_epoch_progress (bool): Show progress bar for epochs within each fold.

    Note:
        The trainer_config.gp_config.optimize_scorer default is False for standalone
        BooleanGP usage, but benchmarking typically benefits from optimization (faster
        evaluation via data deduplication and sample weights). Set optimize_scorer=True
        in your gp_config for better benchmarking performance.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.configs import BooleanGPConfig, TrainerConfig, BenchmarkerConfig
        >>> data = np.array([[True, False], [False, True], [True, True], [False, False]])
        >>> labels = np.array([1, 0, 1, 0])
        >>> def accuracy(p, l): return float((p == l).mean())
        >>> gp_config = BooleanGPConfig(score_fn=accuracy, optimize_scorer=True)
        >>> trainer_config = TrainerConfig(gp_config=gp_config, num_epochs=10)
        >>> config = BenchmarkerConfig(
        ...     data=data, labels=labels, trainer_config=trainer_config, n_folds=2
        ... )
        >>> config.num_runs
        30
        >>> config.n_folds
        2
    """

    data: ndarray
    labels: ndarray
    trainer_config: TrainerConfig
    num_runs: int = 30
    test_size: float = 0.2
    n_folds: int = 5
    n_jobs: int = -1
    base_seed: int = 0
    show_run_progress: bool = True
    show_fold_progress: bool = True
    show_epoch_progress: bool = True


def validate_benchmarker_config(config: BenchmarkerConfig) -> None:
    """
    Validate BenchmarkerConfig.

    Validates the full dataset and benchmarker-specific parameters. The nested
    trainer_config is validated without requiring train_data (since data is
    provided at the benchmarker level and split per fold).

    Args:
        config (BenchmarkerConfig): Configuration to validate.

    Raises:
        TypeError: If any field has incorrect type.
        ValueError: If any field has invalid value.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.configs import BooleanGPConfig, TrainerConfig, BenchmarkerConfig
        >>> from hgp_lib.configs.benchmarker_config import validate_benchmarker_config
        >>> data = np.array([[True, False], [False, True], [True, True], [False, False]])
        >>> labels = np.array([1, 0, 1, 0])
        >>> def accuracy(p, l): return float((p == l).mean())
        >>> gp_config = BooleanGPConfig(score_fn=accuracy)
        >>> trainer_config = TrainerConfig(gp_config=gp_config, num_epochs=10)
        >>> config = BenchmarkerConfig(
        ...     data=data, labels=labels, trainer_config=trainer_config, n_folds=2
        ... )
        >>> validate_benchmarker_config(config)  # No error
    """
    # Validate full dataset
    check_X_y(config.data, config.labels)

    # Validate trainer config (without requiring data in gp_config)
    validate_trainer_config(config.trainer_config, require_data=False)

    # Validate benchmarker-specific parameters
    check_isinstance(config.num_runs, int)
    if config.num_runs < 1:
        raise ValueError("num_runs must be a positive integer")
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    check_isinstance(config.n_folds, int)
    if config.n_folds < 2:
        raise ValueError("n_folds must be at least 2")
