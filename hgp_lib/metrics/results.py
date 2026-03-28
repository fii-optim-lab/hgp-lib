"""Result dataclasses for fold, run, and experiment level aggregation."""

from dataclasses import dataclass
from functools import cached_property
from typing import List, Dict

import numpy as np

from . import PopulationHistory
from ..rules import Rule


@dataclass
class RunResult:
    """Result of one complete run with k-fold cross-validation."""

    run_id: int
    seed: int
    best_fold_idx: int
    folds: List[PopulationHistory]
    test_score: float
    test_tp: int
    test_fp: int
    test_fn: int
    test_tn: int
    feature_names: Dict[int, str]

    @cached_property
    def best_fold(self):
        return self.folds[self.best_fold_idx]

    @cached_property
    def best_rule(self) -> Rule:
        """Get best rule from best fold (by validation score)."""
        return self.best_fold.global_best_rule

    @cached_property
    def fold_val_scores(self) -> List[float]:
        """Get best validation score from each fold."""
        return [
            fold.best_val_score
            for fold in self.folds
            if fold.best_val_score is not None
        ]

    @cached_property
    def mean_val_score(self) -> float:
        """Get mean of best validation scores across all folds."""
        scores = self.fold_val_scores
        if len(scores) == 0:
            return 0.0
        return float(np.mean(scores))

    @cached_property
    def train_confusion_matrix(self) -> str:
        best_fold = self.best_fold
        return f"[TP: {best_fold.tp}, FP: {best_fold.fp}, FN: {best_fold.fn}, TN: {best_fold.tn}]"

    @cached_property
    def val_confusion_matrix(self) -> str:
        best_fold = self.best_fold
        if best_fold.val_tp is None:
            return "[]"
        return f"[TP: {best_fold.val_tp}, FP: {best_fold.val_fp}, FN: {best_fold.val_fn}, TN: {best_fold.val_tn}]"

    @cached_property
    def test_confusion_matrix(self) -> str:
        return f"[TP: {self.test_tp}, FP: {self.test_fp}, FN: {self.test_fn}, TN: {self.test_tn}]"


@dataclass
class ExperimentResult:
    """Aggregated results across multiple runs."""

    runs: List[RunResult]

    @cached_property
    def best_run(self) -> RunResult:
        """
        Get the run with the highest mean validation score across folds.

        Returns:
            RunResult with highest mean validation score.
        """
        best_run = None
        best_mean = -float("inf")

        for run in self.runs:
            mean_val = run.mean_val_score
            if mean_val > best_mean:
                best_mean = mean_val
                best_run = run

        return best_run

    @cached_property
    def best_rule(self) -> Rule:
        """
        Get the best rule from the best fold of the best run.

        Best run = run with highest mean validation score across folds.
        Best fold = fold with highest best validation score in best run.
        Best rule = rule with highest validation score in best fold.

        Returns:
            Best Rule.
        """
        best_run = self.best_run
        return best_run.folds[best_run.best_fold_idx].global_best_rule

    @cached_property
    def test_scores(self) -> list[float]:
        """Get test scores from all runs."""
        return [run.test_score for run in self.runs]
