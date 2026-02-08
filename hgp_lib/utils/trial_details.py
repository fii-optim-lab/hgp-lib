"""
Trial details extraction and storage for Optuna integration.

This module provides utilities for extracting detailed trial information from
benchmark results and storing them as Optuna trial user attributes.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from hgp_lib.metrics import BenchmarkResult, RunMetrics


@dataclass
class TrialDetails:
    """
    Aggregated trial details for Optuna storage.

    Contains all score lists, statistics, best run/fold information,
    and rule strings extracted from a benchmark result.

    Attributes:
        all_train_scores: All train scores from all runs, sorted descending.
        all_val_scores: All validation scores from all runs, sorted descending.
        all_test_scores: All test scores from all runs, sorted descending.
        train_stats: Statistics (mean, std, min, max) for train scores.
        val_stats: Statistics (mean, std, min, max) for validation scores.
        test_stats: Statistics (mean, std, min, max) for test scores.
        best_run_idx: Index of the run with highest mean validation score.
        best_fold_idx: Index of the best fold within the best run.
        best_run_train_score: Train score of the best fold in the best run.
        best_run_val_score: Validation score of the best fold in the best run.
        best_run_test_score: Test score of the best run.
        best_rule_str: Human-readable rule string (inline format).
        best_rule_indented: Human-readable rule string (indented format).
        is_hierarchical: Whether the trial used hierarchical GP.
        num_children: Number of child populations (0 if non-hierarchical).
    """

    all_train_scores: List[float]
    all_val_scores: List[float]
    all_test_scores: List[float]
    train_stats: Dict[str, float] = field(default_factory=dict)
    val_stats: Dict[str, float] = field(default_factory=dict)
    test_stats: Dict[str, float] = field(default_factory=dict)
    best_run_idx: int = 0
    best_fold_idx: int = 0
    best_run_train_score: float = 0.0
    best_run_val_score: float = 0.0
    best_run_test_score: float = 0.0
    best_rule_str: str = ""
    best_rule_human_readable: str = ""
    is_hierarchical: bool = False
    num_children: int = 0


def _compute_stats(scores: List[float]) -> Dict[str, float]:
    """
    Compute statistics (mean, std, min, max) for a list of scores.

    Args:
        scores: List of numeric scores.

    Returns:
        Dictionary with keys 'mean', 'std', 'min', 'max'.
    """
    arr = np.array(scores)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _find_best_run_idx(run_metrics: List[RunMetrics]) -> int:
    """
    Find the index of the run with the highest mean validation score.

    Args:
        run_metrics: List of RunMetrics from all runs.

    Returns:
        Index of the best run.
    """
    mean_val_scores = [np.mean(rm.fold_val_scores) for rm in run_metrics]
    return int(np.argmax(mean_val_scores))


def _detect_hierarchical(run_metrics: List[RunMetrics]) -> tuple[bool, int]:
    """
    Detect if the trial used hierarchical GP and count children.

    Args:
        run_metrics: List of RunMetrics from all runs.

    Returns:
        Tuple of (is_hierarchical, num_children).
    """
    for rm in run_metrics:
        if rm.train_history is not None:
            for epoch in rm.train_history.epochs:
                if epoch.children_best_scores is not None:
                    return True, len(epoch.children_best_scores)
    return False, 0


def extract_trial_details(
    result: BenchmarkResult,
) -> TrialDetails:
    """
    Extract all trial details from a benchmark result.

    Extracts all scores from all runs, computes statistics, identifies the
    best run and fold, and formats the best rule with feature names.

    Args:
        result: BenchmarkResult containing all run metrics.

    Returns:
        TrialDetails containing all extracted information.

    Raises:
        ValueError: If result contains no runs.
    """
    if not result.run_metrics:
        raise ValueError("BenchmarkResult contains no runs")

    # Extract all scores from all runs
    all_train_scores = []
    all_val_scores = []
    all_test_scores = []

    for rm in result.run_metrics:
        # Use the best fold's train score for each run
        all_train_scores.append(rm.fold_train_scores[rm.best_fold_idx])
        all_val_scores.append(rm.best_fold_val_score)
        all_test_scores.append(rm.test_score)

    # Sort scores in descending order
    all_train_scores_sorted = sorted(all_train_scores, reverse=True)
    all_val_scores_sorted = sorted(all_val_scores, reverse=True)
    all_test_scores_sorted = sorted(all_test_scores, reverse=True)

    # Compute statistics
    train_stats = _compute_stats(all_train_scores)
    val_stats = _compute_stats(all_val_scores)
    test_stats = _compute_stats(all_test_scores)

    # Find best run (max mean validation score across folds)
    best_run_idx = _find_best_run_idx(result.run_metrics)
    best_run = result.run_metrics[best_run_idx]

    # Best fold within best run
    best_fold_idx = best_run.best_fold_idx

    # Best run scores
    best_run_train_score = best_run.fold_train_scores[best_fold_idx]
    best_run_val_score = best_run.best_fold_val_score
    best_run_test_score = best_run.test_score

    # Format best rule
    best_rule = best_run.best_rule
    best_rule_str = str(best_rule)
    best_rule_human_readable = best_rule.to_str(
        result.feature_names_per_run[best_run_idx], indent=0
    )

    # Detect hierarchical GP
    is_hierarchical, num_children = _detect_hierarchical(result.run_metrics)

    return TrialDetails(
        all_train_scores=all_train_scores_sorted,
        all_val_scores=all_val_scores_sorted,
        all_test_scores=all_test_scores_sorted,
        train_stats=train_stats,
        val_stats=val_stats,
        test_stats=test_stats,
        best_run_idx=best_run_idx,
        best_fold_idx=best_fold_idx,
        best_run_train_score=best_run_train_score,
        best_run_val_score=best_run_val_score,
        best_run_test_score=best_run_test_score,
        best_rule_str=best_rule_str,
        best_rule_human_readable=best_rule_human_readable,
        is_hierarchical=is_hierarchical,
        num_children=num_children,
    )


def store_trial_details(
    trial,  # optuna.Trial - not type-hinted to avoid hard dependency
    details: TrialDetails,
) -> None:
    """
    Store trial details as Optuna trial user attributes.

    All fields are stored in JSON-serializable format. Lists are stored
    directly as they are JSON-serializable.

    Args:
        trial: Optuna trial object to store attributes on.
        details: TrialDetails to store.
    """
    # Store all score lists (already JSON-serializable)
    trial.set_user_attr("all_train_scores", details.all_train_scores)
    trial.set_user_attr("all_val_scores", details.all_val_scores)
    trial.set_user_attr("all_test_scores", details.all_test_scores)

    # Store statistics (flatten dict to individual attributes)
    for stat_name, stat_value in details.train_stats.items():
        trial.set_user_attr(f"train_{stat_name}", stat_value)
    for stat_name, stat_value in details.val_stats.items():
        trial.set_user_attr(f"val_{stat_name}", stat_value)
    for stat_name, stat_value in details.test_stats.items():
        trial.set_user_attr(f"test_{stat_name}", stat_value)

    # Store best run/fold info
    trial.set_user_attr("best_run_idx", details.best_run_idx)
    trial.set_user_attr("best_fold_idx", details.best_fold_idx)
    trial.set_user_attr("best_run_train_score", details.best_run_train_score)
    trial.set_user_attr("best_run_val_score", details.best_run_val_score)
    trial.set_user_attr("best_run_test_score", details.best_run_test_score)

    # Store rule strings
    trial.set_user_attr("best_rule", details.best_rule_str)
    trial.set_user_attr("best_rule_human_readable", details.best_rule_human_readable)

    # Store hierarchical info
    trial.set_user_attr("is_hierarchical", details.is_hierarchical)
    trial.set_user_attr("num_children", details.num_children)


def get_trial_details_from_attrs(
    user_attrs: Dict,
) -> TrialDetails | None:
    """
    Reconstruct TrialDetails from trial user attributes.

    This function handles backward compatibility for trials that may not
    have all attributes (e.g., trials from older versions).

    Args:
        user_attrs: Dictionary of user attributes from an Optuna trial.

    Returns:
        TrialDetails if the trial has the required attributes, None otherwise.
        Returns None for trials without detailed attributes (old trials).
    """
    # Check if this trial has detailed attributes
    if "all_train_scores" not in user_attrs:
        return None

    try:
        return TrialDetails(
            all_train_scores=user_attrs.get("all_train_scores", []),
            all_val_scores=user_attrs.get("all_val_scores", []),
            all_test_scores=user_attrs.get("all_test_scores", []),
            train_stats={
                "mean": user_attrs.get("train_mean", 0.0),
                "std": user_attrs.get("train_std", 0.0),
                "min": user_attrs.get("train_min", 0.0),
                "max": user_attrs.get("train_max", 0.0),
            },
            val_stats={
                "mean": user_attrs.get("val_mean", 0.0),
                "std": user_attrs.get("val_std", 0.0),
                "min": user_attrs.get("val_min", 0.0),
                "max": user_attrs.get("val_max", 0.0),
            },
            test_stats={
                "mean": user_attrs.get("test_mean", 0.0),
                "std": user_attrs.get("test_std", 0.0),
                "min": user_attrs.get("test_min", 0.0),
                "max": user_attrs.get("test_max", 0.0),
            },
            best_run_idx=user_attrs.get("best_run_idx", 0),
            best_fold_idx=user_attrs.get("best_fold_idx", 0),
            best_run_train_score=user_attrs.get("best_run_train_score", 0.0),
            best_run_val_score=user_attrs.get("best_run_val_score", 0.0),
            best_run_test_score=user_attrs.get("best_run_test_score", 0.0),
            best_rule_str=user_attrs.get("best_rule", ""),
            best_rule_indented=user_attrs.get("best_rule_indented", ""),
            is_hierarchical=user_attrs.get("is_hierarchical", False),
            num_children=user_attrs.get("num_children", 0),
        )
    except (KeyError, TypeError):
        return None


def has_trial_details(user_attrs: Dict) -> bool:
    """
    Check if a trial has detailed attributes.

    This is useful for determining whether a trial was created with the
    enhanced trial details feature or is an older trial without artifacts.

    Args:
        user_attrs: Dictionary of user attributes from an Optuna trial.

    Returns:
        True if the trial has detailed attributes, False otherwise.
    """
    return "all_train_scores" in user_attrs and "is_hierarchical" in user_attrs
