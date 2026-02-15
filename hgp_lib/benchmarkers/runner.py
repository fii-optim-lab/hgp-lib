import random
from copy import deepcopy
from dataclasses import replace
from functools import partial
from multiprocessing import Queue
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from ..configs import BenchmarkerConfig
from ..metrics import PopulationHistory, RunResult
from ..trainers import GPTrainer
from ..utils.metrics import optimize_scorer_for_data

from .progress import send_progress


def execute_single_run(
    run_id: int,
    seed: int,
    config: BenchmarkerConfig,
    progress_queue: Optional[Queue] = None,
) -> RunResult:
    """
    Execute one benchmark run: stratified train/test split, per-fold binarization,
    k-fold CV training, best-fold selection, and test-set evaluation.

    This is a module-level function so it can be pickled for `multiprocessing`.

    **Per-fold binarization:** For each fold a fresh `deepcopy` of the configured
    binarizer is fitted on the training fold (with labels, enabling supervised
    binning for numerical features) and used to transform the validation fold.
    After selecting the best fold, its binarizer transforms the held-out test
    data. This prevents data leakage across folds and between train/test sets.

    Args:
        run_id (int): Index of the run (0-based).
        seed (int): Random seed for stratified split and k-fold.
        config (BenchmarkerConfig): Configuration for the benchmark run.
        progress_queue (Queue | None): Optional queue for sending progress updates.

    Returns:
        RunResult: Contains run_id, seed, folds, test_score, best_rule, feature_names.

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

    np.random.seed(seed)
    random.seed(seed)

    train_data, test_data, train_labels, test_labels = train_test_split(
        config.data,
        config.labels,
        test_size=config.test_size,
        stratify=config.labels,
        random_state=seed,
    )

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=seed)

    folds: List[PopulationHistory] = []
    binarizers = []
    feature_names_per_binarizer = []

    fold_splits = list(skf.split(train_data, train_labels))
    if show_folds:
        fold_splits = tqdm(fold_splits, total=config.n_folds, desc="Folds", leave=False)

    epoch_callback = (
        partial(send_progress, progress_queue, "epoch") if use_queue else None
    )

    best_fold_idx = 0
    best_val_score = -float("inf")
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        fold_train = train_data.iloc[train_idx]
        fold_train_labels = train_labels[train_idx]

        binarizer = deepcopy(config.binarizer)
        fold_train = binarizer.fit_transform(fold_train, fold_train_labels)
        binarizers.append(binarizer)
        feature_names_per_binarizer.append(
            {i: col for i, col in enumerate(fold_train.columns)}
        )
        fold_train = fold_train.to_numpy(dtype=bool)

        fold_val = binarizer.transform(train_data.iloc[val_idx]).to_numpy(dtype=bool)
        fold_val_labels = train_labels[val_idx]

        fold_gp_config = replace(
            gp_template,
            train_data=fold_train,
            train_labels=fold_train_labels,
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
        history = trainer.fit()
        if history.best_val_score is not None and history.best_val_score > best_val_score:
            best_val_score = history.best_val_score
            best_fold_idx = fold_idx

        folds.append(history)

        send_progress(progress_queue, "fold", 1)

    del train_data, train_labels

    best_rule = folds[best_fold_idx].global_best_rule

    # Transform test data using best fold's binarizer
    binarizer_here = binarizers[best_fold_idx]
    feature_names = feature_names_per_binarizer[best_fold_idx]
    test_data = binarizer_here.transform(test_data).to_numpy(dtype=bool)

    if gp_template.optimize_scorer:
        test_score_fn, test_data_opt, test_labels_opt = optimize_scorer_for_data(
            gp_template.score_fn, test_data, test_labels
        )
    else:
        test_score_fn = gp_template.score_fn
        test_data_opt = test_data
        test_labels_opt = test_labels

    test_score = float(test_score_fn(best_rule.evaluate(test_data_opt), test_labels_opt))

    send_progress(progress_queue, "run", 1)

    return RunResult(
        run_id=run_id,
        seed=seed,
        best_fold_idx=best_fold_idx,
        folds=folds,
        test_score=test_score,
        feature_names=feature_names,
    )


def single_run_wrapper(
    args: Tuple[int, int, BenchmarkerConfig, Optional[Queue]],
) -> RunResult:
    """
    Picklable wrapper for multiprocessing Pool.

    Args:
        args (tuple): Tuple of (run_id, seed, config, progress_queue).

    Returns:
        RunResult: Result for the run.
    """
    run_id, seed, config, progress_queue = args
    return execute_single_run(run_id, seed, config, progress_queue)
