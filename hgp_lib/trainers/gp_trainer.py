from typing import Callable, List

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from ..algorithms import BooleanGP
from ..configs import TrainerConfig, validate_trainer_config
from ..metrics import EpochMetrics, TrainerResult, TrainingHistory, ValidateBestMetrics
from ..utils.metrics import optimize_scorer_for_data
from ..utils.validation import check_X_y


class GPTrainer:
    """
    High-level trainer for Boolean Genetic Programming.
    Accepts a TrainerConfig containing a BooleanGPConfig and training options.
    Runs the training loop and optionally validates every val_every epochs.
    Returns a TrainerResult with TrainingHistory (per-epoch best/mean/std) and
    optional validation history.
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
        >>> result = trainer.fit()
        >>> test_metrics = trainer.score(val_data, val_labels)
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

    def fit(self) -> TrainerResult:
        """
        Trains the Boolean GP model for the specified number of epochs.
        Returns:
            TrainerResult: train_history (TrainingHistory), val_history (TrainingHistory | None),
                best_rule (Rule), best_score (float).
        """
        train_epochs: List[EpochMetrics] = []
        val_epochs: List[EpochMetrics] = []
        val_best = 0.0

        with tqdm(
            range(self.num_epochs),
            desc="Epochs",
            disable=not self.progress_bar,
            leave=self.leave_progress_bar,
        ) as tbar:
            for epoch in tbar:
                step_metrics = self.gp_algo.step()

                # Extract children scores for hierarchical GP
                children_best_scores = None
                if step_metrics.children_metrics:
                    children_best_scores = [
                        child.current_best for child in step_metrics.children_metrics
                    ]

                train_epochs.append(
                    EpochMetrics(
                        epoch=step_metrics.epoch,
                        best_score=step_metrics.current_best,
                        mean_score=step_metrics.mean_score,
                        std_score=step_metrics.std_score,
                        best_rule=step_metrics.best_rule,
                        regenerated=step_metrics.regenerated,
                        children_best_scores=children_best_scores,
                    )
                )

                if (
                    self.progress_callback is not None
                    and (epoch + 1) % self.config.progress_update_interval == 0
                ):
                    self.progress_callback(self.config.progress_update_interval)

                if self.val_data is not None and (epoch + 1) % self.val_every == 0:
                    val_metrics = self.gp_algo.validate_population(
                        self.val_data,
                        self.val_labels,
                        score_fn=self.val_score_fn,
                    )
                    val_best = val_metrics.best
                    pop_scores = val_metrics.population_scores
                    val_epochs.append(
                        EpochMetrics(
                            epoch=epoch,
                            best_score=val_best,
                            mean_score=float(np.mean(pop_scores)),
                            std_score=float(np.std(pop_scores)),
                            best_rule=val_metrics.best_rule,
                            regenerated=False,
                        )
                    )

                tbar.set_postfix(
                    {
                        "current_best": f"{step_metrics.current_best:.4f}",
                        "val_best": f"{val_best:.4f}",
                    }
                )

        # Send remaining epochs not covered by progress_update_interval
        remaining_epochs = self.num_epochs % self.config.progress_update_interval
        if remaining_epochs > 0 and self.progress_callback is not None:
            self.progress_callback(remaining_epochs)

        train_history = TrainingHistory(epochs=train_epochs)
        val_history = TrainingHistory(epochs=val_epochs)
        best_rule = self.gp_algo.real_best_rule
        best_score = self.gp_algo.real_best_score

        return TrainerResult(
            train_history=train_history,
            val_history=val_history,
            best_rule=best_rule,
            best_score=best_score,
        )

    def score(
        self,
        test_data: ndarray,
        test_labels: ndarray,
        score_fn: Callable[[ndarray, ndarray], float] | None = None,
        all_time_best: bool = True,
    ) -> ValidateBestMetrics:
        """
        Evaluates the trained model on test data.
        Args:
            test_data (ndarray): Test data (2D boolean array).
            test_labels (ndarray): Test labels (1D integer array).
            score_fn (Callable | None): Optional; uses trainer's _original_score_fn if None. Default: `None`.
            all_time_best (bool): If True, evaluate all-time best rule. Default: `True`.
        Returns:
            ValidateBestMetrics: best (float) and best_rule (Rule).
        """
        check_X_y(test_data, test_labels)
        fn = score_fn if score_fn is not None else self._original_score_fn
        return self.gp_algo.validate_best(
            test_data, test_labels, score_fn=fn, all_time_best=all_time_best
        )
