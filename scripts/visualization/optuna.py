import logging

import numpy as np
import optuna

import os
import tempfile

import matplotlib

import matplotlib.pyplot as plt

from optuna.artifacts import upload_artifact

from hgp_lib.metrics.results import ExperimentResult
from .plots import (
    plot_experiment_boxplots,
    plot_best_fold_generations,
    plot_all_folds_val_scores,
    plot_population_bands,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def store_trial_attributes(
    trial: optuna.Trial,
    result: ExperimentResult,
) -> None:
    """
    Store experiment metrics as Optuna trial user attributes.

    Attributes are prefixed for alphabetical grouping in Optuna dashboard:
    - 01_train_* : Training scores (mean of best train scores per fold, per run)
    - 02_val_* : Validation scores (mean of best val scores per fold, per run)
    - 03_test_* : Test scores (one per run)
    - 04_best_* : Best run/fold indices

    Args:
        trial: Optuna trial object.
        result: ExperimentResult from benchmarker.
    """
    all_train_scores = []
    all_val_scores = []

    for run in result.runs:
        # Train: mean of best train score per fold
        train_scores_per_fold = []
        for fold in run.folds:
            best_train = max(gen.best_train_score for gen in fold.generations)
            train_scores_per_fold.append(best_train)
        all_train_scores.append(float(np.mean(train_scores_per_fold)))

        # Val: mean of best val score per fold
        val_scores_per_fold = run.fold_val_scores
        if val_scores_per_fold:
            all_val_scores.append(float(np.mean(val_scores_per_fold)))

    # 01_train_*
    trial.set_user_attr("01_train_scores", all_train_scores)
    trial.set_user_attr("01_train_mean", float(np.mean(all_train_scores)))
    trial.set_user_attr("01_train_std", float(np.std(all_train_scores)))

    # 02_val_*
    if all_val_scores:
        trial.set_user_attr("02_val_scores", all_val_scores)
        trial.set_user_attr("02_val_mean", float(np.mean(all_val_scores)))
        trial.set_user_attr("02_val_std", float(np.std(all_val_scores)))

    # 03_test_*
    test_scores = result.test_scores
    trial.set_user_attr("03_test_scores", test_scores)
    trial.set_user_attr("03_test_mean", float(np.mean(test_scores)))
    trial.set_user_attr("03_test_std", float(np.std(test_scores)))

    # 04_best_*
    best_run = result.best_run
    trial.set_user_attr("04_best_run_idx", best_run.run_id)
    trial.set_user_attr("04_best_fold_idx", best_run.best_fold_idx)
    trial.set_user_attr("04_best_test_score", best_run.test_score)

    # 05_rule_*
    best_rule = result.best_rule
    trial.set_user_attr("05_rule_string", str(best_rule))
    trial.set_user_attr(
        "05_rule_human_readable", best_rule.to_str(best_run.feature_names)
    )
    trial.set_user_attr("05_rule_complexity", len(best_rule))

    # 06_confusion
    trial.set_user_attr(
        "06_train_cms", [run.train_confusion_matrix for run in result.runs]
    )
    trial.set_user_attr("06_val_cms", [run.val_confusion_matrix for run in result.runs])
    trial.set_user_attr(
        "06_test_cms", [run.test_confusion_matrix for run in result.runs]
    )


def upload_trial_artifacts(
    trial: optuna.Trial,
    result: ExperimentResult,
    artifact_store,
    top_k_transfer: int = 10,
) -> None:
    """
    Generate plots and upload to Optuna artifact store.

    Args:
        trial: Optuna trial object.
        result: ExperimentResult from benchmarker.
        artifact_store: Optuna FileSystemArtifactStore.
        top_k_transfer: top_k_transfer from BooleanGPConfig, used for
            population bands plot.
    """
    plots = [
        ("experiment_boxplots", plot_experiment_boxplots, (result,)),
        ("best_fold_generations", plot_best_fold_generations, (result,)),
        ("all_folds_val_scores", plot_all_folds_val_scores, (result,)),
        ("population_bands", plot_population_bands, (result, top_k_transfer)),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for plot_name, plot_fn, args in plots:
            try:
                fig = plot_fn(*args)
                if fig is None:
                    continue

                plot_path = os.path.join(tmpdir, f"{plot_name}.png")
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                # TODO: FutureWarning: upload_artifact() got {'study_or_trial', 'artifact_store', 'file_path'} as positional arguments but they were expected to be given as keyword arguments.
                # ositional arguments ['study_or_trial', 'file_path', 'artifact_store'] in upload_artifact() have been deprecated since v4.0.0. They will be replaced with the corresponding keyword arguments in v6.0.0, so please use the keyword specification
                upload_artifact(trial, plot_path, artifact_store)

            except Exception as e:
                logger.warning(
                    f"Trial {trial.number}: Failed to generate {plot_name}: {e}"
                )
                continue
