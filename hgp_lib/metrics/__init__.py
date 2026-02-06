from dataclasses import dataclass
from typing import List

from numpy import ndarray

from ..rules import Rule


@dataclass
class EpochMetrics:
    """
    Metrics for a single training epoch.
    Attributes:
        epoch (int): Epoch number (0-indexed).
        best_score (float): Best fitness score in this epoch.
        mean_score (float): Mean fitness score of the population.
        std_score (float): Standard deviation of population scores.
        best_rule (Rule): Best rule in this epoch.
        regenerated (bool): Whether population was regenerated this step.
        children_best_scores (List[float] | None): Best scores from each child
            population at this epoch. Only populated for hierarchical GP runs
            where num_child_populations > 0. None for non-hierarchical runs.
    """

    epoch: int
    best_score: float
    mean_score: float
    std_score: float
    best_rule: Rule
    regenerated: bool = False
    children_best_scores: List[float] | None = None


@dataclass
class TrainingHistory:
    """
    Accumulated training metrics across epochs.
    Attributes:
        epochs (List[EpochMetrics]): Per-epoch metrics.
    Examples:
        >>> from hgp_lib.metrics import TrainingHistory, EpochMetrics
        >>> from hgp_lib.rules import Literal
        >>> rule = Literal(value=0)
        >>> epochs = [
        ...     EpochMetrics(epoch=0, best_score=0.5, mean_score=0.4, std_score=0.1, best_rule=rule),
        ...     EpochMetrics(epoch=1, best_score=0.6, mean_score=0.5, std_score=0.08, best_rule=rule),
        ... ]
        >>> history = TrainingHistory(epochs=epochs)
        >>> history.best_scores()
        [0.5, 0.6]
        >>> history.mean_scores()
        [0.4, 0.5]
    """

    epochs: List[EpochMetrics]

    # TODO: Each method should have doctests.

    def best_scores(self) -> List[float]:
        """Return list of best scores per epoch."""
        return [e.best_score for e in self.epochs]

    def mean_scores(self) -> List[float]:
        """Return list of mean scores per epoch."""
        return [e.mean_score for e in self.epochs]

    def std_scores(self) -> List[float]:
        """Return list of std scores per epoch."""
        return [e.std_score for e in self.epochs]


@dataclass
class StepMetrics:
    """
    Metrics returned by a single training step.
    Attributes:
        best (float): Best fitness score in current run.
        best_rule (Rule): Best rule in current run.
        real_best (float): All-time best fitness score.
        real_best_rule (Rule): All-time best rule.
        current_best (float): Best score in current generation.
        mean_score (float): Mean fitness score of the population.
        std_score (float): Standard deviation of population scores.
        population_scores (ndarray): Fitness scores for all rules.
        epoch (int): Current epoch number (0-indexed).
        best_not_improved_epochs (int): Epochs since last improvement.
        regenerated (bool): Whether population was regenerated this step.
        children_metrics (List[StepMetrics]): Metrics for each child population.
    """

    best: float
    best_rule: Rule
    real_best: float
    real_best_rule: Rule
    current_best: float
    mean_score: float
    std_score: float
    population_scores: ndarray
    epoch: int
    best_not_improved_epochs: int
    regenerated: bool
    children_metrics: List["StepMetrics"]


@dataclass
class ValidateBestMetrics:
    """
    Metrics returned by validate_best.
    Attributes:
        best (float): Fitness score of the best rule on the data.
        best_rule (Rule): The evaluated rule.
    """

    best: float
    best_rule: Rule


@dataclass
class ValidatePopulationMetrics:
    """
    Metrics returned by validate_population.
    Attributes:
        best (float): Fitness score of the best rule on the data.
        best_rule (Rule): The evaluated best rule.
        population_scores (ndarray): Fitness scores for all rules in the population.
    """

    best: float
    best_rule: Rule
    population_scores: ndarray


@dataclass
class TrainerResult:
    """
    Metrics returned by the trainer after fitting.
    Attributes:
        train_history (TrainingHistory): Per-epoch training metrics.
        val_history (TrainingHistory | None): Per-validation-step metrics, or None if no validation.
        best_rule (Rule): Best rule found during training.
        best_score (float): Best fitness score achieved.
    """

    train_history: TrainingHistory
    val_history: TrainingHistory | None
    best_rule: Rule
    best_score: float


@dataclass
class RunMetrics:
    """
    Metrics returned by a single benchmark run.
    Attributes:
        run_id (int): Index of the run (0-based).
        seed (int): Random seed used for this run.
        fold_train_scores (List[float]): Training score per fold.
        fold_val_scores (List[float]): Validation score per fold.
        best_fold_idx (int): Index of the fold with best validation score.
        best_fold_val_score (float): Best validation score across folds.
        test_score (float): Test set score of the selected best rule.
        best_rule (Rule): The best rule selected from the best fold.
        fold_train_histories (List[TrainingHistory] | None): Epoch-level training
            metrics for each fold. None if not captured.
        fold_val_histories (List[TrainingHistory | None] | None): Epoch-level
            validation metrics for each fold. Inner None if no validation was
            performed for that fold. Outer None if not captured.
    """

    run_id: int
    seed: int
    fold_train_scores: List[float]
    fold_val_scores: List[float]
    best_fold_idx: int
    best_fold_val_score: float
    test_score: float
    best_rule: Rule
    fold_train_histories: List[TrainingHistory] | None = None
    fold_val_histories: List[TrainingHistory | None] | None = None

    @property
    def train_history(self) -> TrainingHistory | None:
        """Training history for the best fold."""
        if self.fold_train_histories is None:
            return None
        return self.fold_train_histories[self.best_fold_idx]

    @property
    def val_history(self) -> TrainingHistory | None:
        """Validation history for the best fold."""
        if self.fold_val_histories is None:
            return None
        return self.fold_val_histories[self.best_fold_idx]


@dataclass
class BenchmarkResult:
    """
    Metrics returned by the benchmarker after fitting.
    Attributes:
        run_metrics (List[RunMetrics]): Per-run details.
        mean_test_score (float): Mean test score across runs.
        std_test_score (float): Standard deviation of test scores.
        mean_best_val_score (float): Mean best validation score across runs.
        std_best_val_score (float): Standard deviation of best validation scores.
        all_test_scores (List[float]): Test score for each run.
        all_best_rules (List[Rule]): Best rule from each run.
    """

    run_metrics: List[RunMetrics]
    mean_test_score: float
    std_test_score: float
    mean_best_val_score: float
    std_best_val_score: float
    all_test_scores: List[float]
    all_best_rules: List[Rule]


__all__ = [
    "EpochMetrics",
    "TrainingHistory",
    "StepMetrics",
    "ValidateBestMetrics",
    "ValidatePopulationMetrics",
    "TrainerResult",
    "RunMetrics",
    "BenchmarkResult",
]
