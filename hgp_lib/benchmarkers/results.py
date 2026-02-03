from typing import List

import numpy as np

from hgp_lib.metrics import BenchmarkResult, RunMetrics


def aggregate_results(run_metrics: List[RunMetrics]) -> BenchmarkResult:
    """
    Aggregate per-run metrics into BenchmarkResult.

    Args:
        run_metrics (List[RunMetrics]): Metrics from each run.

    Returns:
        BenchmarkResult: Aggregated metrics containing mean/std of test and
        validation scores, plus the full list of per-run metrics.

    Examples:
        >>> from hgp_lib.benchmarkers.results import aggregate_results
        >>> from hgp_lib.metrics import RunMetrics
        >>> from hgp_lib.rules import Literal
        >>> rule = Literal(value=0)
        >>> metrics = [
        ...     RunMetrics(run_id=0, seed=0, fold_train_scores=[0.7], fold_val_scores=[0.75],
        ...                best_fold_idx=0, test_score=0.8, best_fold_val_score=0.75, best_rule=rule),
        ...     RunMetrics(run_id=1, seed=1, fold_train_scores=[0.8], fold_val_scores=[0.85],
        ...                best_fold_idx=0, test_score=0.9, best_fold_val_score=0.85, best_rule=rule),
        ... ]
        >>> result = aggregate_results(metrics)
        >>> round(result.mean_test_score, 2)
        0.85
        >>> len(result.all_best_rules)
        2
    """
    test_scores = [m.test_score for m in run_metrics]
    best_val_scores = [m.best_fold_val_score for m in run_metrics]
    all_best_rules = [m.best_rule for m in run_metrics]

    return BenchmarkResult(
        run_metrics=run_metrics,
        mean_test_score=float(np.mean(test_scores)),
        std_test_score=float(np.std(test_scores)),
        mean_best_val_score=float(np.mean(best_val_scores)),
        std_best_val_score=float(np.std(best_val_scores)),
        all_test_scores=test_scores,
        all_best_rules=all_best_rules,
    )
