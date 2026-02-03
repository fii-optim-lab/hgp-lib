import os
import random
from typing import Any, Callable, Dict, List

import numpy as np
from numpy import ndarray
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ..crossover import CrossoverExecutor
from ..metrics import BenchmarkMetrics, RunMetrics
from ..mutations import MutationExecutor
from ..populations import PopulationGenerator
from ..rules import Rule
from ..selections import BaseSelection
from ..trainers import GPTrainer
from ..utils.metrics import optimize_scorer_for_data
from ..utils.validation import (
    check_isinstance,
    validate_callable,
    validate_trainer_params,
)


def _execute_single_run(
    run_id: int,
    seed: int,
    config: Dict[str, Any],
) -> RunMetrics:
    """
    Execute one benchmark run: stratified train/test split, k-fold CV, select best fold, test.

    This is a module-level function so it can be pickled for process_map.

    Args:
        run_id (int):
            Index of the run (0-based).
        seed (int):
            Random seed for stratified split and k-fold.
        config (Dict[str, Any]):
            Configuration dictionary containing all parameters for the run.
            Must include: data, labels, test_size, n_folds, num_epochs, score_fn,
            val_score_fn, check_valid, population_generator, mutation_executor,
            crossover_executor, selection, regeneration, regeneration_patience,
            val_every, show_fold_progress, show_epoch_progress, progress_bar,
            optimize_scorer.

    Returns:
        RunMetrics: Metrics for the run including fold scores, best rule, and validation score.

    Raises:
        RuntimeError: If no best rule is available after training a fold.
    """
    data = config["data"]
    labels = config["labels"]
    test_size = config["test_size"]
    n_folds = config["n_folds"]
    num_epochs = config["num_epochs"]
    score_fn = config["score_fn"]
    val_score_fn = config["val_score_fn"]
    check_valid = config["check_valid"]
    population_generator = config["population_generator"]
    mutation_executor = config["mutation_executor"]
    crossover_executor = config["crossover_executor"]
    selection = config["selection"]
    regeneration = config["regeneration"]
    regeneration_patience = config["regeneration_patience"]
    val_every = config["val_every"]
    show_fold_progress = config["show_fold_progress"]
    show_epoch_progress = config["show_epoch_progress"]
    progress_bar = config["progress_bar"]
    optimize_scorer = config["optimize_scorer"]

    np.random.seed(seed)
    random.seed(seed)

    train_data, test_data, train_labels, test_labels = train_test_split(
        data,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_train_scores: List[float] = []
    fold_val_scores: List[float] = []
    fold_best_rules: List[Rule] = []

    fold_iterator = skf.split(train_data, train_labels)
    if show_fold_progress and progress_bar:
        fold_iterator = tqdm(
            list(fold_iterator),
            total=n_folds,
            desc="  Folds",
            leave=False,
        )

    for train_idx, val_idx in fold_iterator:
        fold_train = train_data[train_idx]
        fold_val = train_data[val_idx]
        fold_train_labels = train_labels[train_idx]
        fold_val_labels = train_labels[val_idx]

        # Optimize scorers for this fold's data if requested
        if optimize_scorer:
            fold_score_fn, fold_train, fold_train_labels = optimize_scorer_for_data(
                score_fn, fold_train, fold_train_labels
            )
            fold_val_score_fn, fold_val, fold_val_labels = optimize_scorer_for_data(
                val_score_fn, fold_val, fold_val_labels
            )
        else:
            fold_score_fn = score_fn
            fold_val_score_fn = val_score_fn

        trainer = GPTrainer(
            score_fn=fold_score_fn,
            num_epochs=num_epochs,
            train_data=fold_train,
            train_labels=fold_train_labels,
            val_data=fold_val,
            val_labels=fold_val_labels,
            val_score_fn=fold_val_score_fn,
            check_valid=check_valid,
            population_generator=population_generator,
            mutation_executor=mutation_executor,
            crossover_executor=crossover_executor,
            selection=selection,
            regeneration=regeneration,
            regeneration_patience=regeneration_patience,
            val_every=val_every,
            progress_bar=show_epoch_progress and progress_bar,
            progress_desc="    Epochs"
            if (show_epoch_progress and progress_bar)
            else None,
        )
        trainer.fit()

        train_metrics = trainer.gp_algo.validate_best(
            fold_train, fold_train_labels, score_fn=fold_score_fn, all_time_best=True
        )
        val_metrics = trainer.gp_algo.validate_best(
            fold_val, fold_val_labels, score_fn=fold_val_score_fn, all_time_best=True
        )

        fold_train_scores.append(float(train_metrics["best"]))
        fold_val_scores.append(float(val_metrics["best"]))

        if trainer.gp_algo.real_best_rule is None:
            raise RuntimeError(
                f"No best rule available after training fold. "
                f"Run {run_id}, seed {seed}."
            )
        fold_best_rules.append(trainer.gp_algo.real_best_rule.copy())

    best_fold_idx = int(np.argmax(fold_val_scores))
    best_fold_val_score = fold_val_scores[best_fold_idx]
    best_rule = fold_best_rules[best_fold_idx]

    # Optimize scorer for test data if requested
    if optimize_scorer:
        test_score_fn, test_data_opt, test_labels_opt = optimize_scorer_for_data(
            score_fn, test_data, test_labels
        )
    else:
        test_score_fn = score_fn
        test_data_opt = test_data
        test_labels_opt = test_labels

    test_predictions = best_rule.evaluate(test_data_opt)
    test_score = float(test_score_fn(test_predictions, test_labels_opt))

    return RunMetrics(
        run_id=run_id,
        seed=seed,
        fold_train_scores=fold_train_scores,
        fold_val_scores=fold_val_scores,
        best_fold_idx=best_fold_idx,
        best_fold_val_score=best_fold_val_score,
        test_score=test_score,
        best_rule=best_rule,
    )


def _single_run_wrapper(args: tuple) -> RunMetrics:
    """
    Picklable wrapper for process_map.

    Args:
        args (tuple): Tuple of (run_id, seed, config).

    Returns:
        RunMetrics: Metrics for the run.
    """
    run_id, seed, config = args
    return _execute_single_run(run_id, seed, config)


class GPBenchmarker:
    """
    Benchmarker for Boolean GP: runs multiple full runs with stratified train/test
    split and k-fold CV per run, then aggregates results.

    Each run uses a different random seed for the stratified split. Within each
    run, k-fold cross-validation is performed on the training set; the best rule
    is selected from the fold with the best validation score and evaluated on
    the held-out test set.

    Args:
        score_fn (Callable[[ndarray, ndarray], float]):
            Function that computes fitness scores. Signature: `score_fn(predictions, labels) -> float`.
            Higher scores indicate better fitness. When using parallel execution (`n_jobs != 1`),
            `score_fn` must be picklable (use module-level functions, not lambdas or closures).
            If the scorer supports a `sample_weight` parameter and `optimize_scorer=True`,
            the scorer will be optimized per fold by deduplicating data and using sample weights.
        num_epochs (int):
            Number of training epochs per fold.
        data (ndarray):
            Full dataset as a 2D boolean array with instances on rows and features on columns.
        labels (ndarray):
            Labels as a 1D integer array (0 or 1 for binary classification).
        num_runs (int, optional):
            Number of benchmark runs. Default: `30`.
        test_size (float, optional):
            Fraction for held-out validation set (0 < test_size < 1). Default: `0.2`.
        n_folds (int, optional):
            Number of folds for k-fold CV. Default: `5`.
        n_jobs (int, optional):
            Number of parallel jobs. -1 means all CPUs, 1 means sequential. Default: `-1`.
        base_seed (int, optional):
            Base random seed. Run i uses seed = base_seed + i. Default: `0`.
        optimize_scorer (bool, optional):
            Whether to optimize the scorer for each data split by removing duplicate
            rows and using sample weights. This can significantly speed up scoring for
            datasets with many duplicate rows. Requires the scorer to accept a
            `sample_weight` parameter. If `False`, the scorer is used as-is. Default: `True`.

            **Important**: Do NOT pass pre-optimized scorers (e.g., from `optimize_scorer_for_data`)
            when `optimize_scorer=True`. Pre-optimized scorers have sample weights bound to
            the original data, which become invalid after train/test/fold splits. Either:

            - Pass a base scorer and let the benchmarker optimize it per split (`optimize_scorer=True`), or
            - Pass a base scorer without optimization (`optimize_scorer=False`).
        val_score_fn (Callable[[ndarray, ndarray], float] | None, optional):
            Optional validation scorer. If `None`, uses `score_fn`. Same optimization
            rules apply as for `score_fn`. Default: `None`.
        check_valid (Callable[[Rule], bool] | None, optional):
            Optional rule validator for mutation/crossover. Default: `None`.
        population_generator (PopulationGenerator | None, optional):
            Generator for initial population. If `None`, GPTrainer creates a default. Default: `None`.
        mutation_executor (MutationExecutor | None, optional):
            Executor for mutations. If `None`, GPTrainer creates a default. Default: `None`.
        crossover_executor (CrossoverExecutor | None, optional):
            Executor for crossover. If `None`, GPTrainer creates a default. Default: `None`.
        selection (BaseSelection | None, optional):
            Selection strategy. If `None`, GPTrainer creates a default. Default: `None`.
        regeneration (bool, optional):
            Whether to regenerate population on plateau. Default: `False`.
        regeneration_patience (int, optional):
            Epochs without improvement before regeneration. Default: `100`.
        val_every (int, optional):
            Validation frequency in epochs. Default: `100`.
        progress_bar (bool, optional):
            Master switch for all progress bars. Default: `True`.
        show_run_progress (bool, optional):
            Show progress bar for runs. Default: `True`.
        show_fold_progress (bool, optional):
            Show progress bar for folds. Note: in parallel mode this may cause
            interleaved output. Default: `True`.
        show_epoch_progress (bool, optional):
            Show progress bar for epochs. Note: in parallel mode this may cause
            interleaved output. Default: `True`.

    Examples:
        Basic usage with scorer optimization (recommended)::

            import numpy as np
            from hgp_lib.benchmarkers import GPBenchmarker

            def f1_score(predictions, labels, sample_weight=None):
                # Scorer that supports sample_weight for optimization
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

            benchmarker = GPBenchmarker(
                score_fn=f1_score,  # Base scorer, will be optimized per fold
                num_epochs=100,
                data=data,
                labels=labels,
                optimize_scorer=True,  # Default: optimize per fold
            )
            metrics = benchmarker.fit()

        Without scorer optimization::

            benchmarker = GPBenchmarker(
                score_fn=simple_accuracy,  # Scorer without sample_weight support
                num_epochs=100,
                data=data,
                labels=labels,
                optimize_scorer=False,  # Don't try to optimize
            )
    """

    def __init__(
        self,
        score_fn: Callable[[ndarray, ndarray], float],
        num_epochs: int,
        data: ndarray,
        labels: ndarray,
        *,
        num_runs: int = 30,
        test_size: float = 0.2,
        n_folds: int = 5,
        n_jobs: int = -1,
        base_seed: int = 0,
        optimize_scorer: bool = True,
        val_score_fn: Callable[[ndarray, ndarray], float] | None = None,
        check_valid: Callable[[Rule], bool] | None = None,
        population_generator: PopulationGenerator | None = None,
        mutation_executor: MutationExecutor | None = None,
        crossover_executor: CrossoverExecutor | None = None,
        selection: BaseSelection | None = None,
        regeneration: bool = False,
        regeneration_patience: int = 100,
        val_every: int = 100,
        progress_bar: bool = True,
        show_run_progress: bool = True,
        show_fold_progress: bool = True,
        show_epoch_progress: bool = True,
    ):
        # Validate common trainer parameters
        validate_trainer_params(
            score_fn=score_fn,
            num_epochs=num_epochs,
            train_data=data,
            train_labels=labels,
            val_every=val_every,
            regeneration_patience=regeneration_patience,
            val_score_fn=val_score_fn,
        )

        # Validate benchmarker-specific parameters
        check_isinstance(num_runs, int)
        check_isinstance(test_size, float)
        check_isinstance(n_folds, int)
        check_isinstance(n_jobs, int)
        check_isinstance(base_seed, int)
        check_isinstance(optimize_scorer, bool)
        check_isinstance(regeneration, bool)
        check_isinstance(progress_bar, bool)
        check_isinstance(show_run_progress, bool)
        check_isinstance(show_fold_progress, bool)
        check_isinstance(show_epoch_progress, bool)

        if num_runs < 1:
            raise ValueError("num_runs must be a positive integer")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be in (0, 1)")
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")

        if check_valid is not None:
            validate_callable(check_valid)

        if population_generator is not None:
            check_isinstance(population_generator, PopulationGenerator)
        if mutation_executor is not None:
            check_isinstance(mutation_executor, MutationExecutor)
        if crossover_executor is not None:
            check_isinstance(crossover_executor, CrossoverExecutor)
        if selection is not None:
            check_isinstance(selection, BaseSelection)

        self.score_fn = score_fn
        self.val_score_fn = val_score_fn if val_score_fn is not None else score_fn
        self.num_epochs = num_epochs
        self.data = data
        self.labels = labels
        self.num_runs = num_runs
        self.test_size = test_size
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.base_seed = base_seed
        self.optimize_scorer = optimize_scorer
        self.check_valid = check_valid
        self.population_generator = population_generator
        self.mutation_executor = mutation_executor
        self.crossover_executor = crossover_executor
        self.selection = selection
        self.regeneration = regeneration
        self.regeneration_patience = regeneration_patience
        self.val_every = val_every
        self.progress_bar = progress_bar
        self.show_run_progress = show_run_progress
        self.show_fold_progress = show_fold_progress
        self.show_epoch_progress = show_epoch_progress

        self._run_metrics: List[RunMetrics] | None = None

    def _effective_n_jobs(self) -> int:
        """
        Compute the effective number of parallel jobs to use.

        Resolves the `n_jobs` parameter to an actual worker count:
        - If `n_jobs < 0`, returns the number of CPUs (or 1 if detection fails).
        - If `n_jobs >= 1`, returns `n_jobs`.
        - If `n_jobs == 0`, returns 1 (treated as sequential).

        Returns:
            int: The number of parallel workers to use (always >= 1).

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.benchmarkers import GPBenchmarker
            >>> data = np.array([[True, False], [False, True], [True, True], [False, False]])
            >>> labels = np.array([1, 0, 1, 0])
            >>> def score(p, l): return float((p == l).mean())
            >>> b = GPBenchmarker(score, 1, data, labels, n_jobs=4, optimize_scorer=False)
            >>> b._effective_n_jobs()
            4
            >>> b = GPBenchmarker(score, 1, data, labels, n_jobs=1, optimize_scorer=False)
            >>> b._effective_n_jobs()
            1
            >>> b = GPBenchmarker(score, 1, data, labels, n_jobs=0, optimize_scorer=False)
            >>> b._effective_n_jobs()
            1
        """
        if self.n_jobs < 0:
            return os.cpu_count() or 1
        return max(1, self.n_jobs)

    def _build_config(self) -> Dict[str, Any]:
        """
        Build a picklable config dict for worker processes.

        Returns:
            Dict[str, Any]: Configuration dictionary with all parameters needed
            by `_execute_single_run`.

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.benchmarkers import GPBenchmarker
            >>> data = np.array([[True, False], [False, True], [True, True], [False, False]])
            >>> labels = np.array([1, 0, 1, 0])
            >>> def score(p, l): return float((p == l).mean())
            >>> b = GPBenchmarker(score, 1, data, labels, n_folds=2, optimize_scorer=False)
            >>> config = b._build_config()
            >>> config["n_folds"]
            2
            >>> config["optimize_scorer"]
            False
            >>> config["score_fn"] is score
            True
        """
        return {
            "data": self.data,
            "labels": self.labels,
            "test_size": self.test_size,
            "n_folds": self.n_folds,
            "num_epochs": self.num_epochs,
            "score_fn": self.score_fn,
            "val_score_fn": self.val_score_fn,
            "optimize_scorer": self.optimize_scorer,
            "check_valid": self.check_valid,
            "population_generator": self.population_generator,
            "mutation_executor": self.mutation_executor,
            "crossover_executor": self.crossover_executor,
            "selection": self.selection,
            "regeneration": self.regeneration,
            "regeneration_patience": self.regeneration_patience,
            "val_every": self.val_every,
            "show_fold_progress": self.show_fold_progress,
            "show_epoch_progress": self.show_epoch_progress,
            "progress_bar": self.progress_bar,
        }

    def _run_sequential(self, config: Dict[str, Any]) -> List[RunMetrics]:
        """
        Run all benchmark runs sequentially with nested progress bars.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the run.

        Returns:
            List[RunMetrics]: Metrics for each run.
        """
        run_metrics: List[RunMetrics] = []

        run_iterator = range(self.num_runs)
        if self.progress_bar and self.show_run_progress:
            run_iterator = tqdm(
                run_iterator,
                total=self.num_runs,
                desc="Benchmark Runs",
            )

        for run_id in run_iterator:
            seed = self.base_seed + run_id
            metrics = _execute_single_run(run_id, seed, config)
            run_metrics.append(metrics)

        return run_metrics

    def _aggregate_results(self, run_metrics: List[RunMetrics]) -> BenchmarkMetrics:
        """
        Aggregate per-run metrics into BenchmarkMetrics.

        Args:
            run_metrics (List[RunMetrics]): Metrics from each run.

        Returns:
            BenchmarkMetrics: Aggregated metrics containing mean/std of test and
            validation scores, plus the full list of per-run metrics.

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.benchmarkers import GPBenchmarker
            >>> data = np.array([[True, False], [False, True], [True, True], [False, False]])
            >>> labels = np.array([1, 0, 1, 0])
            >>> def score(p, l): return float((p == l).mean())
            >>> b = GPBenchmarker(score, 1, data, labels, optimize_scorer=False)
            >>> mock_metrics = [
            ...     {"test_score": 0.8, "best_fold_val_score": 0.75, "run_id": 0,
            ...      "seed": 0, "fold_train_scores": [0.7], "fold_val_scores": [0.75],
            ...      "best_fold_idx": 0, "best_rule": None},
            ...     {"test_score": 0.9, "best_fold_val_score": 0.85, "run_id": 1,
            ...      "seed": 1, "fold_train_scores": [0.8], "fold_val_scores": [0.85],
            ...      "best_fold_idx": 0, "best_rule": None},
            ... ]
            >>> result = b._aggregate_results(mock_metrics)
            >>> round(result["mean_test_score"], 2)
            0.85
            >>> round(result["std_test_score"], 2)
            0.05
            >>> round(result["mean_best_val_score"], 2)
            0.8
            >>> len(result["all_test_scores"])
            2
            >>> len(result["all_best_rules"])
            2
        """
        test_scores = [m["test_score"] for m in run_metrics]
        best_val_scores = [m["best_fold_val_score"] for m in run_metrics]
        all_best_rules = [m["best_rule"] for m in run_metrics]

        return BenchmarkMetrics(
            run_metrics=run_metrics,
            mean_test_score=float(np.mean(test_scores)),
            std_test_score=float(np.std(test_scores)),
            mean_best_val_score=float(np.mean(best_val_scores)),
            std_best_val_score=float(np.std(best_val_scores)),
            all_test_scores=test_scores,
            all_best_rules=all_best_rules,
        )

    def fit(self) -> BenchmarkMetrics:
        """
        Run all benchmark runs (parallel or sequential) and aggregate results.

        Executes `num_runs` independent benchmark runs. Each run performs:
        1. Stratified train/test split using the run's seed.
        2. K-fold cross-validation on the training set.
        3. Selection of the best rule from the fold with highest validation score.
        4. Evaluation of the best rule on the held-out test set.

        Results are aggregated across all runs to compute mean and standard
        deviation of test scores and validation scores.

        Returns:
            BenchmarkMetrics: Dictionary containing:
                - ``run_metrics`` (List[RunMetrics]): Per-run detailed metrics.
                - ``mean_test_score`` (float): Mean test score across all runs.
                - ``std_test_score`` (float): Standard deviation of test scores.
                - ``mean_best_val_score`` (float): Mean best validation score.
                - ``std_best_val_score`` (float): Std of best validation scores.
                - ``all_test_scores`` (List[float]): Test score for each run.
                - ``all_best_rules`` (List[Rule]): Best rule from each run.
        """
        effective_n_jobs = self._effective_n_jobs()
        config = self._build_config()

        if effective_n_jobs == 1:
            run_metrics = self._run_sequential(config)
        else:
            run_args = [
                (run_id, self.base_seed + run_id, config)
                for run_id in range(self.num_runs)
            ]
            run_metrics = process_map(
                _single_run_wrapper,
                run_args,
                max_workers=effective_n_jobs,
                desc="Benchmark Runs",
                disable=not self.progress_bar or not self.show_run_progress,
            )

        self._run_metrics = run_metrics
        return self._aggregate_results(run_metrics)
