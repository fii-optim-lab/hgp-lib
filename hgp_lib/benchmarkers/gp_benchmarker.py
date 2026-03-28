import multiprocessing
import os
from typing import List

from tqdm import tqdm

from ..configs import BenchmarkerConfig, validate_benchmarker_config
from ..metrics import ExperimentResult, RunResult

from .progress import ProgressConfig, ProgressListener
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

    Args:
        config (BenchmarkerConfig): Configuration with data,
            labels, binarizer, trainer_config template, and benchmarker-specific
            options. See `BenchmarkerConfig` for more details.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from hgp_lib.configs import BooleanGPConfig, TrainerConfig, BenchmarkerConfig
        >>> from hgp_lib.benchmarkers import GPBenchmarker
        >>> from hgp_lib.rules import Rule
        >>> data = pd.DataFrame({
        ...     "f1": [True, False, True, False, True, False, True, False],
        ...     "f2": [False, True, True, False, False, True, True, False],
        ... })
        >>> labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        >>> def acc(p, l): return float((p == l).mean())
        >>> gp_config = BooleanGPConfig(score_fn=acc, optimize_scorer=False)
        >>> trainer_config = TrainerConfig(gp_config=gp_config, num_epochs=3, progress_bar=False)
        >>> config = BenchmarkerConfig(
        ...     data=data, labels=labels, trainer_config=trainer_config,
        ...     num_runs=2, n_folds=2, n_jobs=1,
        ... )
        >>> benchmarker = GPBenchmarker(config)
        >>> result = benchmarker.fit()
        >>> len(result.runs)
        2
        >>> isinstance(result.best_rule, Rule)
        True
    """

    def __init__(self, config: BenchmarkerConfig):
        validate_benchmarker_config(config)
        self.config = config
        if self.config.binarizer is None:
            self.config.binarizer = StandardBinarizer()
        self._run_results: ExperimentResult | None = None

    def _effective_n_jobs(self) -> int:
        """
        Compute the effective number of parallel jobs to use.

        Returns:
            int: The number of parallel workers to use (always >= 1).

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from hgp_lib.configs import BooleanGPConfig, TrainerConfig, BenchmarkerConfig
            >>> from hgp_lib.benchmarkers import GPBenchmarker
            >>> data = pd.DataFrame({"f": [True, False, True, False, True, False, True, False]})
            >>> labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
            >>> def acc(p, l): return float((p == l).mean())
            >>> gp = BooleanGPConfig(score_fn=acc, optimize_scorer=False)
            >>> tc = TrainerConfig(gp_config=gp, num_epochs=1, progress_bar=False)
            >>> cfg = BenchmarkerConfig(data=data, labels=labels, trainer_config=tc,
            ...     num_runs=3, n_folds=2, n_jobs=1)
            >>> GPBenchmarker(cfg)._effective_n_jobs()
            1
        """
        n_jobs = self.config.n_jobs
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        return max(1, min(n_jobs, self.config.num_runs))

    def _run_sequential(self) -> ExperimentResult:
        """Run all benchmark runs sequentially with nested progress bars."""
        run_results: List[RunResult] = []

        if self.config.show_run_progress:
            self.config.trainer_config.leave_progress_bar = False

        for run_id in tqdm(
            range(self.config.num_runs),
            desc="Benchmark Runs",
            disable=not self.config.show_run_progress,
        ):
            result = execute_single_run(
                run_id, self.config.base_seed + run_id, self.config
            )
            run_results.append(result)

        return ExperimentResult(runs=run_results)

    def _run_parallel(self, n_jobs: int) -> ExperimentResult:
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
                run_results = pool.map(single_run_wrapper, run_args)
            # Normal completion - wait for listener to finish processing
            listener.join()
        except Exception:
            # Error occurred - force stop the listener to prevent hang
            listener.stop()
            raise

        return ExperimentResult(runs=run_results)

    def fit(self) -> ExperimentResult:
        """
        Run all benchmark runs (parallel or sequential) and aggregate results.

        Each run performs a stratified train/test split, k-fold CV with per-fold
        binarization, and test-set evaluation. Results across runs are aggregated
        into an ExperimentResult.

        Returns:
            ExperimentResult: Contains all run results with methods to get
                best_run, best_rule, test scores statistics, etc.
        """
        effective_n_jobs = self._effective_n_jobs()

        if effective_n_jobs == 1:
            run_results = self._run_sequential()
        else:
            run_results = self._run_parallel(effective_n_jobs)

        self._run_results = run_results
        return run_results
