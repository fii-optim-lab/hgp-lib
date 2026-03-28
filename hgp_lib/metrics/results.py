"""Result dataclasses for fold, run, and experiment level aggregation."""

from dataclasses import dataclass
from functools import cached_property
from typing import List, Dict

import numpy as np

from . import PopulationHistory
from ..rules import Rule


@dataclass
class RunResult:
    """
    Result of one complete benchmark run with k-fold cross-validation.

    Contains per-fold training histories, the test-set evaluation of the best
    fold's rule, and the confusion matrix on the held-out test set.

    Args:
        run_id (int): Zero-based index of this run.
        seed (int): Random seed used for the stratified split and k-fold.
        best_fold_idx (int): Index of the fold with the highest validation score.
        folds (List[PopulationHistory]): Training history for each fold.
        test_score (float): Score of the best rule on the held-out test set.
        test_tp (int): True positives on the test set.
        test_fp (int): False positives on the test set.
        test_fn (int): False negatives on the test set.
        test_tn (int): True negatives on the test set.
        feature_names (Dict[int, str]): Mapping from feature index to column name
            (from the binarizer fitted on the best fold).

    Examples:
        >>> from hgp_lib.metrics import RunResult, PopulationHistory
        >>> from hgp_lib.rules import Literal
        >>> fold = PopulationHistory(
        ...     global_best_rule=Literal(value=0), tp=3, fp=1, fn=0, tn=6,
        ... )
        >>> run = RunResult(
        ...     run_id=0, seed=42, best_fold_idx=0, folds=[fold],
        ...     test_score=0.85, test_tp=4, test_fp=1, test_fn=1, test_tn=4,
        ...     feature_names={0: "age", 1: "income"},
        ... )
        >>> run.best_rule
        0
        >>> run.test_confusion_matrix
        '[TP: 4, FP: 1, FN: 1, TN: 4]'
    """

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
    def best_fold(self) -> PopulationHistory:
        """
        The ``PopulationHistory`` of the fold with the highest validation score.

        Examples:
            >>> from hgp_lib.metrics import RunResult, PopulationHistory
            >>> from hgp_lib.rules import Literal
            >>> f0 = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=1, fp=0, fn=0, tn=1,
            ... )
            >>> f1 = PopulationHistory(
            ...     global_best_rule=Literal(value=1), tp=2, fp=0, fn=0, tn=2,
            ... )
            >>> run = RunResult(
            ...     run_id=0, seed=0, best_fold_idx=1, folds=[f0, f1],
            ...     test_score=0.9, test_tp=1, test_fp=0, test_fn=0, test_tn=1,
            ...     feature_names={},
            ... )
            >>> run.best_fold is f1
            True
        """
        return self.folds[self.best_fold_idx]

    @cached_property
    def best_rule(self) -> Rule:
        """
        The global best rule from the best fold.

        Examples:
            >>> from hgp_lib.metrics import RunResult, PopulationHistory
            >>> from hgp_lib.rules import Literal
            >>> fold = PopulationHistory(
            ...     global_best_rule=Literal(value=5), tp=0, fp=0, fn=0, tn=0,
            ... )
            >>> run = RunResult(
            ...     run_id=0, seed=0, best_fold_idx=0, folds=[fold],
            ...     test_score=0.8, test_tp=0, test_fp=0, test_fn=0, test_tn=0,
            ...     feature_names={},
            ... )
            >>> str(run.best_rule)
            '5'
        """
        return self.best_fold.global_best_rule

    @cached_property
    def fold_val_scores(self) -> List[float]:
        """
        Best validation score from each fold (folds without validation are excluded).

        Examples:
            >>> from dataclasses import replace
            >>> from hgp_lib.metrics import RunResult, PopulationHistory, GenerationMetrics
            >>> from hgp_lib.rules import Literal
            >>> g = GenerationMetrics.from_population(
            ...     best_idx=0, best_rule=Literal(value=0),
            ...     train_scores=[0.8], complexities=[1],
            ...     child_population_generation_metrics=[],
            ... )
            >>> g_val = replace(g, val_score=0.7)
            >>> f0 = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
            ...     generations=[g_val],
            ... )
            >>> f1 = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
            ...     generations=[g],
            ... )
            >>> run = RunResult(
            ...     run_id=0, seed=0, best_fold_idx=0, folds=[f0, f1],
            ...     test_score=0.8, test_tp=0, test_fp=0, test_fn=0, test_tn=0,
            ...     feature_names={},
            ... )
            >>> run.fold_val_scores
            [0.7]
        """
        return [
            fold.best_val_score
            for fold in self.folds
            if fold.best_val_score is not None
        ]

    @cached_property
    def fold_train_scores(self) -> List[float]:
        """
        Best training score from each fold (folds without generations are excluded).

        Examples:
            >>> from hgp_lib.metrics import RunResult, PopulationHistory, GenerationMetrics
            >>> from hgp_lib.rules import Literal
            >>> g = GenerationMetrics.from_population(
            ...     best_idx=0, best_rule=Literal(value=0),
            ...     train_scores=[0.8], complexities=[1],
            ...     child_population_generation_metrics=[],
            ... )
            >>> fold = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
            ...     generations=[g],
            ... )
            >>> run = RunResult(
            ...     run_id=0, seed=0, best_fold_idx=0, folds=[fold],
            ...     test_score=0.8, test_tp=0, test_fp=0, test_fn=0, test_tn=0,
            ...     feature_names={},
            ... )
            >>> run.fold_train_scores
            [0.8]
        """
        return [
            fold.best_train_score
            for fold in self.folds
            if fold.best_train_score is not None
        ]

    @cached_property
    def mean_val_score(self) -> float:
        """
        Mean of the best validation scores across all folds. Returns ``0.0`` if no
        fold has a validation score.

        Examples:
            >>> from hgp_lib.metrics import RunResult, PopulationHistory
            >>> from hgp_lib.rules import Literal
            >>> fold = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
            ... )
            >>> run = RunResult(
            ...     run_id=0, seed=0, best_fold_idx=0, folds=[fold],
            ...     test_score=0.8, test_tp=0, test_fp=0, test_fn=0, test_tn=0,
            ...     feature_names={},
            ... )
            >>> run.mean_val_score
            0.0
        """
        scores = self.fold_val_scores
        if len(scores) == 0:
            return 0.0
        return float(np.mean(scores))

    @cached_property
    def mean_train_score(self) -> float:
        """
        Mean of the best training scores across all folds. Returns ``0.0`` if no
        fold has training generations.

        Examples:
            >>> from hgp_lib.metrics import RunResult, PopulationHistory, GenerationMetrics
            >>> from hgp_lib.rules import Literal
            >>> g = GenerationMetrics.from_population(
            ...     best_idx=0, best_rule=Literal(value=0),
            ...     train_scores=[0.85], complexities=[1],
            ...     child_population_generation_metrics=[],
            ... )
            >>> fold = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
            ...     generations=[g],
            ... )
            >>> run = RunResult(
            ...     run_id=0, seed=0, best_fold_idx=0, folds=[fold],
            ...     test_score=0.8, test_tp=0, test_fp=0, test_fn=0, test_tn=0,
            ...     feature_names={},
            ... )
            >>> run.mean_train_score
            0.85
        """
        scores = self.fold_train_scores
        if len(scores) == 0:
            return 0.0
        return float(np.mean(scores))

    @cached_property
    def train_confusion_matrix(self) -> str:
        """
        Formatted confusion matrix string for the best fold's training data.

        Examples:
            >>> from hgp_lib.metrics import RunResult, PopulationHistory
            >>> from hgp_lib.rules import Literal
            >>> fold = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=3, fp=1, fn=2, tn=4,
            ... )
            >>> run = RunResult(
            ...     run_id=0, seed=0, best_fold_idx=0, folds=[fold],
            ...     test_score=0.8, test_tp=0, test_fp=0, test_fn=0, test_tn=0,
            ...     feature_names={},
            ... )
            >>> run.train_confusion_matrix
            '[TP: 3, FP: 1, FN: 2, TN: 4]'
        """
        best_fold = self.best_fold
        return f"[TP: {best_fold.tp}, FP: {best_fold.fp}, FN: {best_fold.fn}, TN: {best_fold.tn}]"

    @cached_property
    def val_confusion_matrix(self) -> str:
        """
        Formatted confusion matrix string for the best fold's validation data.
        Returns ``"[]"`` if no validation data was used.

        Examples:
            >>> from hgp_lib.metrics import RunResult, PopulationHistory
            >>> from hgp_lib.rules import Literal
            >>> fold = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
            ... )
            >>> run = RunResult(
            ...     run_id=0, seed=0, best_fold_idx=0, folds=[fold],
            ...     test_score=0.8, test_tp=0, test_fp=0, test_fn=0, test_tn=0,
            ...     feature_names={},
            ... )
            >>> run.val_confusion_matrix
            '[]'
        """
        best_fold = self.best_fold
        if best_fold.val_tp is None:
            return "[]"
        return f"[TP: {best_fold.val_tp}, FP: {best_fold.val_fp}, FN: {best_fold.val_fn}, TN: {best_fold.val_tn}]"

    @cached_property
    def test_confusion_matrix(self) -> str:
        """
        Formatted confusion matrix string for the held-out test set.

        Examples:
            >>> from hgp_lib.metrics import RunResult, PopulationHistory
            >>> from hgp_lib.rules import Literal
            >>> fold = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
            ... )
            >>> run = RunResult(
            ...     run_id=0, seed=0, best_fold_idx=0, folds=[fold],
            ...     test_score=0.8, test_tp=5, test_fp=2, test_fn=1, test_tn=7,
            ...     feature_names={},
            ... )
            >>> run.test_confusion_matrix
            '[TP: 5, FP: 2, FN: 1, TN: 7]'
        """
        return f"[TP: {self.test_tp}, FP: {self.test_fp}, FN: {self.test_fn}, TN: {self.test_tn}]"


@dataclass
class ExperimentResult:
    """
    Aggregated results across multiple benchmark runs.

    Args:
        runs (List[RunResult]): Results from each independent run.

    Examples:
        >>> from dataclasses import replace
        >>> from hgp_lib.metrics import ExperimentResult, RunResult, PopulationHistory, GenerationMetrics
        >>> from hgp_lib.rules import Literal
        >>> g = GenerationMetrics.from_population(
        ...     best_idx=0, best_rule=Literal(value=0),
        ...     train_scores=[0.8], complexities=[1],
        ...     child_population_generation_metrics=[],
        ... )
        >>> g_low = replace(g, val_score=0.5)
        >>> g_high = replace(g, val_score=0.9)
        >>> fold_low = PopulationHistory(
        ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
        ...     generations=[g_low],
        ... )
        >>> fold_high = PopulationHistory(
        ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
        ...     generations=[g_high],
        ... )
        >>> r1 = RunResult(
        ...     run_id=0, seed=0, best_fold_idx=0, folds=[fold_low],
        ...     test_score=0.8, test_tp=0, test_fp=0, test_fn=0, test_tn=0,
        ...     feature_names={},
        ... )
        >>> r2 = RunResult(
        ...     run_id=1, seed=1, best_fold_idx=0, folds=[fold_high],
        ...     test_score=0.9, test_tp=0, test_fp=0, test_fn=0, test_tn=0,
        ...     feature_names={},
        ... )
        >>> exp = ExperimentResult(runs=[r1, r2])
        >>> exp.test_scores
        [0.8, 0.9]
        >>> exp.best_run.run_id
        1
    """

    runs: List[RunResult]

    @cached_property
    def best_run(self) -> RunResult:
        """
        The run with the highest mean validation score across folds. When no run
        has validation scores, falls back to mean training score.

        Returns:
            RunResult: The best-performing run.

        Examples:
            >>> from hgp_lib.metrics import ExperimentResult, RunResult, PopulationHistory, GenerationMetrics
            >>> from hgp_lib.rules import Literal
            >>> g_low = GenerationMetrics.from_population(
            ...     best_idx=0, best_rule=Literal(value=0),
            ...     train_scores=[0.5], complexities=[1],
            ...     child_population_generation_metrics=[],
            ... )
            >>> g_high = GenerationMetrics.from_population(
            ...     best_idx=0, best_rule=Literal(value=0),
            ...     train_scores=[0.9], complexities=[1],
            ...     child_population_generation_metrics=[],
            ... )
            >>> f_low = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
            ...     generations=[g_low],
            ... )
            >>> f_high = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
            ...     generations=[g_high],
            ... )
            >>> r1 = RunResult(
            ...     run_id=0, seed=0, best_fold_idx=0, folds=[f_low],
            ...     test_score=0.7, test_tp=0, test_fp=0, test_fn=0, test_tn=0,
            ...     feature_names={},
            ... )
            >>> r2 = RunResult(
            ...     run_id=1, seed=1, best_fold_idx=0, folds=[f_high],
            ...     test_score=0.9, test_tp=0, test_fp=0, test_fn=0, test_tn=0,
            ...     feature_names={},
            ... )
            >>> ExperimentResult(runs=[r1, r2]).best_run.run_id
            1
        """
        # Use validation scores when available, otherwise fall back to training
        has_val = any(run.mean_val_score > 0.0 for run in self.runs)

        best_run = None
        best_mean = -float("inf")

        for run in self.runs:
            score = run.mean_val_score if has_val else run.mean_train_score
            if score > best_mean:
                best_mean = score
                best_run = run

        return best_run

    @cached_property
    def best_rule(self) -> Rule:
        """
        The best rule from the best fold of the best run.

        Returns:
            Rule: The overall best rule across the entire experiment.

        Examples:
            >>> from hgp_lib.metrics import ExperimentResult, RunResult, PopulationHistory
            >>> from hgp_lib.rules import Literal
            >>> fold = PopulationHistory(
            ...     global_best_rule=Literal(value=7), tp=0, fp=0, fn=0, tn=0,
            ... )
            >>> run = RunResult(
            ...     run_id=0, seed=0, best_fold_idx=0, folds=[fold],
            ...     test_score=0.8, test_tp=0, test_fp=0, test_fn=0, test_tn=0,
            ...     feature_names={},
            ... )
            >>> str(ExperimentResult(runs=[run]).best_rule)
            '7'
        """
        best_run = self.best_run
        return best_run.folds[best_run.best_fold_idx].global_best_rule

    @cached_property
    def test_scores(self) -> list[float]:
        """
        Test scores from all runs.

        Examples:
            >>> from hgp_lib.metrics import ExperimentResult, RunResult, PopulationHistory
            >>> from hgp_lib.rules import Literal
            >>> fold = PopulationHistory(
            ...     global_best_rule=Literal(value=0), tp=0, fp=0, fn=0, tn=0,
            ... )
            >>> r = RunResult(
            ...     run_id=0, seed=0, best_fold_idx=0, folds=[fold],
            ...     test_score=0.75, test_tp=0, test_fp=0, test_fn=0, test_tn=0,
            ...     feature_names={},
            ... )
            >>> ExperimentResult(runs=[r]).test_scores
            [0.75]
        """
        return [run.test_score for run in self.runs]
