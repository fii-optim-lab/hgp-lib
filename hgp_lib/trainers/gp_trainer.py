from typing import List

from tqdm import tqdm

from ..algorithms import BooleanGP
from ..configs import TrainerConfig, validate_trainer_config
from ..metrics import GenerationMetrics, PopulationHistory
from ..utils.metrics import optimize_scorer_for_data


class GPTrainer:
    """
    High-level trainer for Boolean Genetic Programming.
    Accepts a TrainerConfig containing a BooleanGPConfig and training options.
    Runs the training loop and optionally validates every val_every epochs.
    Returns a HierarchicalHistory with GenerationMetrics per epoch.
    Args:
        config (TrainerConfig): Configuration with gp_config (BooleanGPConfig),
            num_epochs, optional val_data/val_labels, val_every, progress options.
    Examples:
        >>> import numpy as np
        >>> from hgp_lib.configs import BooleanGPConfig, TrainerConfig
        >>> from hgp_lib.trainers import GPTrainer
        >>>
        >>> def accuracy(predictions, labels):
        ...     return np.mean(predictions == labels)
        >>>
        >>> train_data = np.array([[True, False, True, False], [False, True, False, True]])
        >>> train_labels = np.array([1, 0])
        >>> val_data = np.array([[True, True, False, False]])
        >>> val_labels = np.array([1])
        >>> gp_config = BooleanGPConfig(
        ...     score_fn=accuracy,
        ...     train_data=train_data,
        ...     train_labels=train_labels,
        ...     optimize_scorer=False,
        ... )
        >>> config = TrainerConfig(
        ...     gp_config=gp_config,
        ...     num_epochs=10,
        ...     val_data=val_data,
        ...     val_labels=val_labels,
        ...     val_every=5,
        ...     progress_bar=False,
        ... )
        >>> trainer = GPTrainer(config)
        >>> trainer_result_history = trainer.fit()
    """

    def __init__(self, config: TrainerConfig):
        validate_trainer_config(config)

        self.config = config
        self.gp_algo = BooleanGP(config.gp_config)
        self.num_epochs = config.num_epochs
        self.val_every = config.val_every
        self.progress_bar = config.progress_bar
        self.leave_progress_bar = config.leave_progress_bar
        self.progress_callback = config.progress_callback

        self.score_fn = self.gp_algo.score_fn  # Maybe optimized
        self._original_score_fn = self.gp_algo._original_score_fn
        if config.val_data is not None and config.gp_config.optimize_scorer:
            self.val_score_fn, self.val_data, self.val_labels = (
                optimize_scorer_for_data(
                    config.gp_config.score_fn, config.val_data, config.val_labels
                )
            )
        else:
            self.val_score_fn = config.gp_config.score_fn
            self.val_data = config.val_data
            self.val_labels = config.val_labels

    def fit(self) -> PopulationHistory:
        """
        Trains the Boolean GP model for the specified number of epochs.
        Returns:
            HierarchicalHistory: History with parent and child population metrics.
        """
        parent_generations: List[GenerationMetrics] = []
        val_score = 0.0

        with tqdm(
            range(self.num_epochs),
            desc="Epochs",
            disable=not self.progress_bar,
            leave=self.leave_progress_bar,
        ) as tbar:
            for epoch in tbar:
                gen_metrics = self.gp_algo.step()

                # Get validation scores if validation data is available
                if self.val_data is not None and (
                    (epoch + 1) % self.val_every == 0 or epoch == self.num_epochs - 1
                ):
                    val_score = self.gp_algo.evaluate_best(
                        self.val_data,
                        self.val_labels,
                        self.val_score_fn,
                    )

                    gen_metrics.val_score = val_score

                parent_generations.append(gen_metrics)

                if (
                    self.progress_callback is not None
                    and (epoch + 1) % self.config.progress_update_interval == 0
                ):
                    self.progress_callback(self.config.progress_update_interval)

                tbar.set_postfix(
                    {
                        "train_best": f"{gen_metrics.best_train_score:.4f}",
                        "val_best": f"{val_score:.4f}",
                    }
                )

        # Send remaining epochs not covered by progress_update_interval
        remaining_epochs = self.num_epochs % self.config.progress_update_interval
        if remaining_epochs > 0 and self.progress_callback is not None:
            self.progress_callback(remaining_epochs)

        return PopulationHistory(
            generations=parent_generations,
            global_best_rule=self.gp_algo.global_best_rule,
        )
