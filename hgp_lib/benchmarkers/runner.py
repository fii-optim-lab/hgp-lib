import random
from dataclasses import replace
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from ..configs import BenchmarkerConfig
from ..metrics import RunMetrics
from ..rules import Rule
from ..trainers import GPTrainer
from ..utils.metrics import optimize_scorer_for_data


def execute_single_run(
    run_id: int,
    seed: int,
    config: BenchmarkerConfig,
) -> RunMetrics:
    """
    Execute one benchmark run: stratified train/test split, k-fold CV, select best fold, test.

    This is a module-level function so it can be pickled for process_map.

    Args:
        run_id (int): Index of the run (0-based).
        seed (int): Random seed for stratified split and k-fold.
        config (BenchmarkerConfig): Configuration for the benchmark run.

    Returns:
        RunMetrics: Metrics for the run including fold scores, best rule, and validation score.

    Raises:
        RuntimeError: If no best rule is available after training a fold.
    """
    trainer_template = config.trainer_config
    gp_template = trainer_template.gp_config

    show_folds = config.show_fold_progress and trainer_template.progress_bar
    show_epochs = config.show_epoch_progress and trainer_template.progress_bar

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

    fold_train_scores: List[float] = []
    fold_val_scores: List[float] = []
    fold_best_rules: List[Rule] = []

    fold_splits = skf.split(train_data, train_labels)
    if show_folds:
        fold_splits = tqdm(fold_splits, total=config.n_folds, desc="Folds", leave=False)

    for train_idx, val_idx in fold_splits:
        fold_train = train_data[train_idx]
        fold_val = train_data[val_idx]
        fold_train_labels = train_labels[train_idx]
        fold_val_labels = train_labels[val_idx]

        # Create fold-specific configs using replace()
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
        )

        trainer = GPTrainer(fold_trainer_config)
        trainer.fit()

        train_metrics = trainer.gp_algo.validate_best(
            fold_train,
            fold_train_labels,
            score_fn=trainer.gp_algo.score_fn,
            all_time_best=True,
        )
        val_metrics = trainer.gp_algo.validate_best(
            fold_val,
            fold_val_labels,
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

    best_fold_idx = int(np.argmax(fold_val_scores))
    best_fold_val_score = fold_val_scores[best_fold_idx]
    best_rule = fold_best_rules[best_fold_idx]

    # Evaluate on test set
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


def single_run_wrapper(args: Tuple[int, int, BenchmarkerConfig]) -> RunMetrics:
    """
    Picklable wrapper for process_map.

    Args:
        args (tuple): Tuple of (run_id, seed, config).

    Returns:
        RunMetrics: Metrics for the run.
    """
    run_id, seed, config = args
    return execute_single_run(run_id, seed, config)
