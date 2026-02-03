import os
from typing import List

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ..configs import BenchmarkerConfig, validate_benchmarker_config
from ..metrics import BenchmarkResult, RunMetrics

from .config import BenchmarkConfig
from .results import aggregate_results
from .runner import execute_single_run, single_run_wrapper


class GPBenchmarker:
    """
    Benchmarker for Boolean GP: runs multiple full runs with stratified train/test
    split and k-fold CV per run, then aggregates results.

    Accepts only a BenchmarkerConfig. Each run uses a different random seed.
    Within each run, k-fold cross-validation is performed on the training set;
    the best rule is selected from the fold with the best validation score and
    evaluated on the held-out test set.

    Args:
        config (BenchmarkerConfig): Configuration with data, labels, score_fn,
            num_epochs, and optional GP components and progress flags.

    Examples:
        Basic usage with scorer optimization::

            import numpy as np
            from hgp_lib.configs import BenchmarkerConfig
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

            config = BenchmarkerConfig(
                data=data,
                labels=labels,
                score_fn=f1_score,
                num_epochs=100,
                optimize_scorer=True,
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

    def _build_config(self) -> BenchmarkConfig:
        """Build a picklable config for worker processes."""
        # TODO: Do we really need to build this? Don't we have already a config? Can't we just upldate what's needed?
        return BenchmarkConfig(
            data=self.config.data,
            labels=self.config.labels,
            test_size=self.config.test_size,
            n_folds=self.config.n_folds,
            num_epochs=self.config.num_epochs,
            score_fn=self.config.score_fn,
            val_score_fn=self.config.val_score_fn or self.config.score_fn,
            optimize_scorer=self.config.optimize_scorer,
            check_valid=self.config.check_valid,
            population_generator=self.config.population_generator,
            mutation_executor=self.config.mutation_executor,
            crossover_executor=self.config.crossover_executor,
            selection=self.config.selection,
            regeneration=self.config.regeneration,
            regeneration_patience=self.config.regeneration_patience,
            val_every=self.config.val_every,
            show_fold_progress=self.config.show_fold_progress,
            show_epoch_progress=self.config.show_epoch_progress,
            progress_bar=self.config.progress_bar,
        )

    def _run_sequential(self, worker_config: BenchmarkConfig) -> List[RunMetrics]:
        """Run all benchmark runs sequentially with nested progress bars."""
        run_metrics: List[RunMetrics] = []

        for run_id in tqdm(
            range(self.config.num_runs),
            desc="Benchmark Runs",
            disable=not (self.config.progress_bar and self.config.show_run_progress),
        ):
            seed = self.config.base_seed + run_id
            metrics = execute_single_run(run_id, seed, worker_config)
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
        worker_config = self._build_config()

        if effective_n_jobs == 1:
            run_metrics = self._run_sequential(worker_config)
        else:
            run_args = [
                (run_id, self.config.base_seed + run_id, worker_config)
                for run_id in range(self.config.num_runs)
            ]
            run_metrics = process_map(
                single_run_wrapper,
                run_args,
                max_workers=effective_n_jobs,
                desc="Benchmark Runs",
                disable=not self.config.progress_bar
                or not self.config.show_run_progress,
            )

        self._run_metrics = run_metrics
        return aggregate_results(run_metrics)
