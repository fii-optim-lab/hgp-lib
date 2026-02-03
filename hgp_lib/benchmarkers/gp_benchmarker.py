import os
from typing import List

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ..configs import BenchmarkerConfig, validate_benchmarker_config
from ..metrics import BenchmarkResult, RunMetrics

from .results import aggregate_results
from .runner import execute_single_run, single_run_wrapper


class GPBenchmarker:
    """
    Benchmarker for Boolean GP: runs multiple full runs with stratified train/test
    split and k-fold CV per run, then aggregates results.

    Accepts a BenchmarkerConfig containing the full dataset, a TrainerConfig template,
    and benchmarker-specific settings. Each run uses a different random seed.
    Within each run, k-fold cross-validation is performed on the training set;
    the best rule is selected from the fold with the best validation score and
    evaluated on the held-out test set.

    Args:
        config (BenchmarkerConfig): Configuration with data, labels, trainer_config
            template, and benchmarker-specific options (num_runs, n_folds, etc.).

    Examples:
        Basic usage with scorer optimization::

            import numpy as np
            from hgp_lib.configs import BooleanGPConfig, TrainerConfig, BenchmarkerConfig
            from hgp_lib.benchmarkers import GPBenchmarker

            def f1_score(predictions, labels, sample_weight=None):
                if sample_weight is None:
                    tp = (predictions & labels).sum()
                    pred_sum, label_sum = predictions.sum(), labels.sum()
                else:
                    tp = np.dot(predictions & labels, sample_weight)
                    pred_sum = np.dot(predictions, sample_weight)
                    label_sum = np.dot(labels, sample_weight)
                if pred_sum == 0 or label_sum == 0:
                    return 1.0 if pred_sum == label_sum == 0 else 0.0
                return 2 * tp / (pred_sum + label_sum)

            # Create template configs (data will be set per fold)
            gp_config = BooleanGPConfig(score_fn=f1_score, optimize_scorer=True)
            trainer_config = TrainerConfig(gp_config=gp_config, num_epochs=100)

            # Create benchmarker config
            config = BenchmarkerConfig(
                data=data,
                labels=labels,
                trainer_config=trainer_config,
            )
            benchmarker = GPBenchmarker(config)
            result = benchmarker.fit()
    """

    def __init__(self, config: BenchmarkerConfig):
        validate_benchmarker_config(config)
        self.config = config
        self._run_metrics: List[RunMetrics] | None = None

    def _effective_n_jobs(self) -> int:
        """
        Compute the effective number of parallel jobs to use.

        Returns:
            int: The number of parallel workers to use (always >= 1).
        """
        if self.config.n_jobs < 0:
            return os.cpu_count() or 1
        return max(1, self.config.n_jobs)

    def _run_sequential(self) -> List[RunMetrics]:
        """Run all benchmark runs sequentially with nested progress bars."""
        run_metrics: List[RunMetrics] = []

        show_run_progress = (
            self.config.show_run_progress and self.config.trainer_config.progress_bar
        )

        for run_id in tqdm(
            range(self.config.num_runs),
            desc="Benchmark Runs",
            disable=not show_run_progress,
        ):
            seed = self.config.base_seed + run_id
            metrics = execute_single_run(run_id, seed, self.config)
            run_metrics.append(metrics)

        return run_metrics

    def fit(self) -> BenchmarkResult:
        """
        Run all benchmark runs (parallel or sequential) and aggregate results.

        Returns:
            BenchmarkResult: run_metrics, mean_test_score, std_test_score,
                mean_best_val_score, std_best_val_score, all_test_scores, all_best_rules.
        """
        effective_n_jobs = self._effective_n_jobs()

        if effective_n_jobs == 1:
            run_metrics = self._run_sequential()
        else:
            show_run_progress = (
                self.config.show_run_progress
                and self.config.trainer_config.progress_bar
            )
            run_args = [
                (run_id, self.config.base_seed + run_id, self.config)
                for run_id in range(self.config.num_runs)
            ]
            run_metrics = process_map(
                single_run_wrapper,
                run_args,
                max_workers=effective_n_jobs,
                desc="Benchmark Runs",
                disable=not show_run_progress,
            )

        self._run_metrics = run_metrics
        return aggregate_results(run_metrics)
