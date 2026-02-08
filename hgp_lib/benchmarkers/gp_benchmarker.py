import multiprocessing
import os
from typing import List

from tqdm import tqdm

from ..configs import BenchmarkerConfig, validate_benchmarker_config
from ..metrics import BenchmarkResult, RunMetrics

from .progress import ProgressConfig, ProgressListener
from .results import aggregate_results
from .runner import execute_single_run, single_run_wrapper
from ..preprocessing import StandardBinarizer


class GPBenchmarker:
    """
    Benchmarker for Boolean GP: runs multiple independent experiments with
    stratified train/test split and k-fold CV per run, then aggregates results.

    Data flow per run:

    1. Stratified train/test split (using `test_size`).
    2. For each of the `n_folds` folds:

       a. A fresh copy of the `binarizer` is fitted on the training fold.
       b. The validation fold is transformed using the same fitted binarizer.
       c. Binarized data is converted to boolean numpy arrays and used for GP
          training.

    3. The best fold (highest validation score) is selected. Its binarizer is
       used to transform the held-out test set for final evaluation.

    Raw (non-binarized) data should be passed as a
    `pandas.DataFrame` in `BenchmarkerConfig.data`. Binarization happens
    internally per fold to prevent data leakage.

    The `BenchmarkResult` includes `feature_names_per_run`, a mapping from
    literal indices to the binarized column names, enabling human-readable
    rule display via `rule.to_str(feature_names)`.

    Args:
        config (BenchmarkerConfig): Configuration with data,
            labels, binarizer, trainer_config template, and benchmarker-specific
            options. See `BenchmarkerConfig` for more details.

    Examples:
        Basic usage with scorer optimization::

            import numpy as np
            import pandas as pd
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

            data = pd.DataFrame(...)  # raw features (bool / categorical / numeric)
            labels = np.array(...)    # 1-D target array

            # Create template configs (data will be binarized and set per fold)
            gp_config = BooleanGPConfig(score_fn=f1_score, optimize_scorer=True)
            trainer_config = TrainerConfig(gp_config=gp_config, num_epochs=100)

            # Create benchmarker config (binarizer=None -> default StandardBinarizer)
            config = BenchmarkerConfig(
                data=data,
                labels=labels,
                trainer_config=trainer_config,
            )
            benchmarker = GPBenchmarker(config)
            result = benchmarker.fit()

            # Display the best rule in human-readable form
            best_idx = np.argmax(result.all_test_scores)
            print(result.all_best_rules[best_idx].to_str(
                result.feature_names_per_run[best_idx]
            ))
    """

    # TODO: This should be a doctest instead of an example.
    def __init__(self, config: BenchmarkerConfig):
        validate_benchmarker_config(config)
        self.config = config
        if self.config.binarizer is None:
            self.config.binarizer = StandardBinarizer()
        self._run_metrics: List[RunMetrics] | None = None

    def _effective_n_jobs(self) -> int:
        """
        Compute the effective number of parallel jobs to use.

        Returns:
            int: The number of parallel workers to use (always >= 1).
        """
        n_jobs = self.config.n_jobs
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        return max(1, min(n_jobs, self.config.num_runs))

    def _run_sequential(self) -> List[RunMetrics]:
        """Run all benchmark runs sequentially with nested progress bars."""
        run_metrics: List[RunMetrics] = []

        if self.config.show_run_progress:
            self.config.trainer_config.leave_progress_bar = False

        for run_id in tqdm(
            range(self.config.num_runs),
            desc="Benchmark Runs",
            disable=not self.config.show_run_progress,
        ):
            metrics = execute_single_run(
                run_id, self.config.base_seed + run_id, self.config
            )
            run_metrics.append(metrics)

        return run_metrics

    def _run_parallel(self, n_jobs: int) -> List[RunMetrics]:
        """Run all benchmark runs in parallel with centralized progress bars."""
        show_progress = self.config.trainer_config.progress_bar

        total_runs = self.config.num_runs
        total_folds = total_runs * self.config.n_folds
        total_epochs = total_folds * self.config.trainer_config.num_epochs

        progress_config = ProgressConfig(
            total_runs=total_runs,
            total_folds=total_folds,
            total_epochs=total_epochs,
            show_run_progress=self.config.show_run_progress and show_progress,
            show_fold_progress=self.config.show_fold_progress and show_progress,
            show_epoch_progress=self.config.show_epoch_progress and show_progress,
        )

        manager = multiprocessing.Manager()
        queue = manager.Queue()

        listener = ProgressListener(queue, progress_config)
        listener.start()

        run_args = [
            (run_id, self.config.base_seed + run_id, self.config, queue)
            for run_id in range(self.config.num_runs)
        ]

        try:
            with multiprocessing.Pool(processes=n_jobs) as pool:
                run_metrics = pool.map(single_run_wrapper, run_args)
            # Normal completion - wait for listener to finish processing
            listener.join()
        except Exception:
            # Error occurred - force stop the listener to prevent hang
            listener.stop()
            raise

        return run_metrics

    def fit(self) -> BenchmarkResult:
        """
        Run all benchmark runs (parallel or sequential) and aggregate results.

        Each run performs a stratified train/test split, k-fold CV with per-fold
        binarization, and test-set evaluation. Results across runs are aggregated
        into mean/std statistics.

        Returns:
            BenchmarkResult: Contains `run_metrics`, `mean_test_score`,
                `std_test_score`, `mean_best_val_score`, `std_best_val_score`,
                `all_test_scores`, `all_best_rules`, and
                `feature_names_per_run`.
                See `BenchmarkResult` for detailed field descriptions.
        """
        effective_n_jobs = self._effective_n_jobs()

        if effective_n_jobs == 1:
            run_metrics = self._run_sequential()
        else:
            run_metrics = self._run_parallel(effective_n_jobs)

        self._run_metrics = run_metrics
        return aggregate_results(run_metrics)
