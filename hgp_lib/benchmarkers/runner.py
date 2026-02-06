from dataclasses import replace
from functools import partial
from multiprocessing import Queue
from typing import List, Optional, Tuple

import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from ..configs import BenchmarkerConfig
from ..metrics import RunMetrics
from ..rules import Rule
from ..trainers import GPTrainer
from ..utils.metrics import optimize_scorer_for_data

from .progress import send_progress


def execute_single_run(
    run_id: int,
    seed: int,
    config: BenchmarkerConfig,
    progress_queue: Optional[Queue] = None,
) -> RunMetrics:
    """
    Execute one benchmark run: stratified train/test split, k-fold CV, select best fold, test.

    This is a module-level function so it can be pickled for process_map.

    Args:
        run_id (int): Index of the run (0-based).
        seed (int): Random seed for stratified split and k-fold.
        config (BenchmarkerConfig): Configuration for the benchmark run.
        progress_queue (Queue | None): Optional queue for sending progress updates
            to the main process. When provided, local progress bars are disabled
            and progress is sent via the queue instead. Default: `None`.

    Returns:
        RunMetrics: Metrics for the run including fold scores, best rule, and validation score.

    Raises:
        RuntimeError: If no best rule is available after training a fold.
    """
    trainer_template = config.trainer_config
    gp_template = trainer_template.gp_config

    use_queue = progress_queue is not None
    show_folds = (
        config.show_fold_progress and trainer_template.progress_bar and not use_queue
    )
    show_epochs = (
        config.show_epoch_progress and trainer_template.progress_bar and not use_queue
    )

    rng = default_rng(seed)

    train_data, test_data, train_labels, test_labels = train_test_split(
        config.data,
        config.labels,
        test_size=config.test_size,
        stratify=config.labels,
        random_state=seed,
    )

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=seed)

    fold_train_scores: List[float] = []
    fold_val_scores: List[float] = []
    fold_best_rules: List[Rule] = []

    fold_splits = skf.split(train_data, train_labels)
    if show_folds:
        fold_splits = tqdm(fold_splits, total=config.n_folds, desc="Folds", leave=False)

    epoch_callback = (
        partial(send_progress, progress_queue, "epoch") if use_queue else None
    )

    fold_seeds = rng.bit_generator.seed_seq.spawn(config.n_folds)

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        fold_train = train_data[train_idx]
        fold_val = train_data[val_idx]
        fold_train_labels = train_labels[train_idx]
        fold_val_labels = train_labels[val_idx]

        fold_gp_config = replace(
            gp_template,
            train_data=fold_train,
            train_labels=fold_train_labels,
            seed=fold_seeds[fold_idx],
        )
        fold_trainer_config = replace(
            trainer_template,
            gp_config=fold_gp_config,
            val_data=fold_val,
            val_labels=fold_val_labels,
            progress_bar=show_epochs,
            progress_callback=epoch_callback,
        )

        trainer = GPTrainer(fold_trainer_config)
        trainer.fit()

        train_metrics = trainer.gp_algo.validate_best(
            trainer.gp_algo.train_data,
            trainer.gp_algo.train_labels,
            score_fn=trainer.gp_algo.score_fn,
            all_time_best=True,
        )
        val_metrics = trainer.gp_algo.validate_best(
            trainer.val_data,
            trainer.val_labels,
            score_fn=trainer.val_score_fn,
            all_time_best=True,
        )

        fold_train_scores.append(float(train_metrics.best))
        fold_val_scores.append(float(val_metrics.best))

        if trainer.gp_algo.real_best_rule is None:
            raise RuntimeError(
                f"No best rule available after training fold. "
                f"Run {run_id}, seed {seed}."
            )
        fold_best_rules.append(trainer.gp_algo.real_best_rule.copy())

        send_progress(progress_queue, "fold", 1)

    best_fold_idx = int(np.argmax(fold_val_scores))
    best_fold_val_score = fold_val_scores[best_fold_idx]
    best_rule = fold_best_rules[best_fold_idx]

    if gp_template.optimize_scorer:
        test_score_fn, test_data_opt, test_labels_opt = optimize_scorer_for_data(
            gp_template.score_fn, test_data, test_labels
        )
    else:
        test_score_fn = gp_template.score_fn
        test_data_opt = test_data
        test_labels_opt = test_labels

    test_predictions = best_rule.evaluate(test_data_opt)
    test_score = float(test_score_fn(test_predictions, test_labels_opt))

    send_progress(progress_queue, "run", 1)

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


def single_run_wrapper(
    args: Tuple[int, int, BenchmarkerConfig, Optional[Queue]],
) -> RunMetrics:
    """
    Picklable wrapper for multiprocessing Pool.

    Args:
        args (tuple): Tuple of (run_id, seed, config, progress_queue).

    Returns:
        RunMetrics: Metrics for the run.
    """
    run_id, seed, config, progress_queue = args
    return execute_single_run(run_id, seed, config, progress_queue)
