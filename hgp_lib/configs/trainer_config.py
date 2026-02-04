from dataclasses import dataclass
from typing import Callable

from numpy import ndarray

from ..utils.validation import check_isinstance, check_X_y
from .boolean_gp_config import BooleanGPConfig, validate_gp_config


@dataclass
class TrainerConfig:
    """
    Configuration for GPTrainer. Wraps BooleanGPConfig.

    Attributes:
        gp_config (BooleanGPConfig): Configuration for the underlying BooleanGP.
        num_epochs (int): Number of training epochs.
        val_data (ndarray | None): Validation data; optional.
        val_labels (ndarray | None): Validation labels; optional.
        val_every (int): Validate every N epochs.
        progress_bar (bool): Whether to show progress bar.
        progress_callback (Callable[[int], None] | None): Optional callback for progress updates.
            Called every `progress_update_interval` epochs with the number of epochs completed.
            Useful for external progress tracking (e.g., multiprocessing progress bars).
        progress_update_interval (int): How often to call progress_callback (in epochs).

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.configs import BooleanGPConfig, TrainerConfig
        >>> data = np.array([[True, False], [False, True], [True, True], [False, False]])
        >>> labels = np.array([1, 0, 1, 0])
        >>> def accuracy(p, l): return float((p == l).mean())
        >>> gp_config = BooleanGPConfig(score_fn=accuracy, train_data=data, train_labels=labels)
        >>> config = TrainerConfig(gp_config=gp_config, num_epochs=10)
        >>> config.num_epochs
        10
        >>> config.val_every
        100
    """

    gp_config: BooleanGPConfig
    num_epochs: int
    val_data: ndarray | None = None
    val_labels: ndarray | None = None
    val_every: int = 100
    progress_bar: bool = True
    progress_callback: Callable[[int], None] | None = None
    progress_update_interval: int = 100


def validate_trainer_config(config: TrainerConfig, require_data: bool = True) -> None:
    """
    Validate TrainerConfig. Also validates the nested BooleanGPConfig.

    Args:
        config (TrainerConfig): Configuration to validate.
        require_data (bool): If True, validates that train_data and train_labels are
            provided in the nested gp_config. Set to False when validating a template
            config inside BenchmarkerConfig. Default: `True`.

    Raises:
        TypeError: If any field has incorrect type.
        ValueError: If any field has invalid value.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.configs import BooleanGPConfig, TrainerConfig
        >>> from hgp_lib.configs.trainer_config import validate_trainer_config
        >>> data = np.array([[True, False], [False, True]])
        >>> labels = np.array([1, 0])
        >>> def accuracy(p, l): return float((p == l).mean())
        >>> gp_config = BooleanGPConfig(score_fn=accuracy, train_data=data, train_labels=labels)
        >>> config = TrainerConfig(gp_config=gp_config, num_epochs=10)
        >>> validate_trainer_config(config)  # No error
    """
    validate_gp_config(config.gp_config, require_data=require_data)
    check_isinstance(config.num_epochs, int)
    if config.num_epochs < 1:
        raise ValueError("num_epochs must be a positive integer")
    check_isinstance(config.val_every, int)
    if config.val_every < 1:
        raise ValueError("val_every must be a positive integer")
    if (config.val_data is None) != (config.val_labels is None):
        raise ValueError(
            "val_data and val_labels must both be provided or both be None"
        )
    if config.val_data is not None:
        check_X_y(config.val_data, config.val_labels)
