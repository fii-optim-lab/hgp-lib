from typing import List, TypedDict

from numpy import ndarray

from ..rules import Rule


class StepMetrics(TypedDict):
    """
    Metrics returned by a single training step.

    Attributes:
        best (float): Best fitness score in current run.
        best_rule (Rule): Best rule in current run.
        real_best (float): All-time best fitness score.
        real_best_rule (Rule): All-time best rule.
        current_best (float): Best score in current generation.
        population_scores (ndarray): Fitness scores for all rules.
        epoch (int): Current epoch number (0-indexed).
        best_not_improved_epochs (int): Epochs since last improvement.
        regenerated (bool): Whether population was regenerated this step.
    """

    best: float
    best_rule: Rule
    real_best: float
    real_best_rule: Rule
    current_best: float
    population_scores: ndarray
    epoch: int
    best_not_improved_epochs: int
    regenerated: bool


class ValidateBestMetrics(TypedDict):
    """
    Metrics returned by validate_best.

    Attributes:
        best (float): Fitness score of the best rule on the data.
        best_rule (Rule): The evaluated rule.
    """

    best: float
    best_rule: Rule


class ValidatePopulationMetrics(TypedDict):
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


class TrainerMetrics(TypedDict):
    """
    Metrics returned by the trainer after fitting.

    Attributes:
        train_best_history (List[float]): Best training scores per epoch.
        val_best_history (List[float]): Best validation scores at each validation step.
        val_epochs (List[int]): Epoch numbers when validation was performed.
    """

    train_best_history: List[float]
    val_best_history: List[float]
    val_epochs: List[int]


class RunMetrics(TypedDict):
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
    """

    run_id: int
    seed: int
    fold_train_scores: List[float]
    fold_val_scores: List[float]
    best_fold_idx: int
    best_fold_val_score: float
    test_score: float
    best_rule: Rule


class BenchmarkMetrics(TypedDict):
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
    "StepMetrics",
    "ValidateBestMetrics",
    "ValidatePopulationMetrics",
    "TrainerMetrics",
    "RunMetrics",
    "BenchmarkMetrics",
]
