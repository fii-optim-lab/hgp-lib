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
