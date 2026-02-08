from dataclasses import dataclass

from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import KBinsDiscretizer

from ..preprocessing import StandardBinarizer
from ..utils.validation import check_isinstance, check_X_y
from .trainer_config import TrainerConfig, validate_trainer_config


@dataclass
class BenchmarkerConfig:
    """
    Configuration for GPBenchmarker. Used for multi-run benchmarking with k-fold CV.

    Contains a TrainerConfig template that specifies all training settings. The benchmarker
    will create TrainerConfig instances for each fold, replacing the data with fold-specific
    train/validation splits.

    The benchmarker handles binarization internally: for each fold, it fits a fresh copy
    of the binarizer on the training fold and transforms validation/test data with it.
    This avoids data leakage (the binarizer never sees validation or test data during fitting).
    Pass raw (non-binarized) data as a `pandas.DataFrame`.

    Attributes:
        data (DataFrame): Full dataset as a `pandas.DataFrame`. The benchmarker will
            binarize it per-fold using the configured `binarizer`, then convert to a
            boolean numpy array for the GP algorithm. Columns can be boolean, categorical,
            or numeric.
        labels (ndarray): Labels for the full dataset (1-D numpy array).
        trainer_config (TrainerConfig): Template configuration for training. The nested
            `gp_config` does not need `train_data`/`train_labels` (they will be set
            per fold by the benchmarker).
        binarizer (StandardBinarizer | KBinsDiscretizer | None): Binarizer to transform
            features into boolean columns. A fresh `deepcopy` is fitted per fold so
            the original stays unfitted. When `None` (default), a
            `StandardBinarizer()` with default settings is used, which applies
            one-hot-encoding to categorical features and decision-tree-based binarization
            (5 bins) to numerical features. The binarizer must **not** be already fitted.
            Default: `None`.
        num_runs (int): Number of benchmark runs with different random seeds. Default: `30`.
        test_size (float): Fraction of data to hold out for testing. Default: `0.2`.
        n_folds (int): Number of folds for k-fold cross-validation. Default: `5`.
        n_jobs (int): Number of parallel jobs (`-1` = all CPUs, `1` = sequential).
            Default: `-1`.
        base_seed (int): Base random seed; each run uses `base_seed + run_id`.
            Default: `0`.
        show_run_progress (bool): Show progress bar for runs. Default: `True`.
        show_fold_progress (bool): Show progress bar for folds within each run.
            Default: `True`.
        show_epoch_progress (bool): Show progress bar for epochs within each fold.
            Default: `True`.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from hgp_lib.configs import BooleanGPConfig, TrainerConfig, BenchmarkerConfig
        >>> data = pd.DataFrame({
        ...     'feature1': [True, False, True, False],
        ...     'feature2': [False, True, True, False],
        ... })
        >>> labels = np.array([1, 0, 1, 0])
        >>> def accuracy(p, l): return float((p == l).mean())
        >>> gp_config = BooleanGPConfig(score_fn=accuracy)
        >>> trainer_config = TrainerConfig(gp_config=gp_config, num_epochs=10)
        >>> config = BenchmarkerConfig(
        ...     data=data, labels=labels, trainer_config=trainer_config, n_folds=2
        ... )
        >>> config.num_runs
        30
        >>> config.n_folds
        2
    """

    data: DataFrame
    labels: ndarray
    trainer_config: TrainerConfig
    binarizer: StandardBinarizer | KBinsDiscretizer | None = None
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
        >>> import pandas as pd
        >>> from hgp_lib.configs import BooleanGPConfig, TrainerConfig, BenchmarkerConfig
        >>> from hgp_lib.configs.benchmarker_config import validate_benchmarker_config
        >>> data = pd.DataFrame({
        ...     'feature1': [True, False, True, False],
        ...     'feature2': [False, True, True, False],
        ... })
        >>> labels = np.array([1, 0, 1, 0])
        >>> def accuracy(p, l): return float((p == l).mean())
        >>> gp_config = BooleanGPConfig(score_fn=accuracy)
        >>> trainer_config = TrainerConfig(gp_config=gp_config, num_epochs=10)
        >>> config = BenchmarkerConfig(
        ...     data=data, labels=labels, trainer_config=trainer_config, n_folds=2
        ... )
        >>> validate_benchmarker_config(config)  # No error
    """
    check_X_y(config.data, config.labels, x_type=DataFrame)

    # Validate trainer config (without requiring data in gp_config)
    validate_trainer_config(config.trainer_config, require_data=False)

    # TODO: Test that KBinsDiscretizer works end-to-end with the benchmarker runner.
    #       The runner currently calls binarizer.fit_transform(X, y) and .transform(X)
    #       and expects DataFrame outputs with .columns and .to_numpy(). KBinsDiscretizer
    #       returns numpy arrays, so an adapter or protocol may be needed.
    if config.binarizer is not None:
        check_isinstance(config.binarizer, (StandardBinarizer, KBinsDiscretizer))
        if hasattr(config.binarizer, "_is_fitted") and config.binarizer._is_fitted:
            raise ValueError(
                "binarizer must not be fitted before passing to BenchmarkerConfig"
            )
    check_isinstance(config.num_runs, int)
    if config.num_runs < 1:
        raise ValueError("num_runs must be a positive integer")
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    check_isinstance(config.n_folds, int)
    if config.n_folds < 2:
        raise ValueError("n_folds must be at least 2")
