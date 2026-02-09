#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning Script for Boolean GP and Hierarchical GP.

This script uses Optuna to optimize hyperparameters for the BooleanGP algorithm.
It integrates with GPBenchmarker for statistically robust evaluation (30 runs, k-fold CV).
Results are stored in SQLite and viewable via Optuna Dashboard.

Usage:
    python scripts/optuna_hypertuning.py --data-path data/PaySim.hdf --n-trials 100

View results:
    optuna-dashboard sqlite:///optuna_study.db --artifact-dir ./artifacts
"""

import argparse
import logging
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import matplotlib
import numpy as np
import optuna
import pandas as pd
from numpy import ndarray
from optuna.artifacts import FileSystemArtifactStore, upload_artifact

from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.mutations import MutationExecutorFactory
from hgp_lib.populations import (
    CombinedSamplingStrategy,
    FeatureSamplingStrategy,
    InstanceSamplingStrategy,
    PopulationGeneratorFactory,
)
from hgp_lib.preprocessing import StandardBinarizer
from hgp_lib.selections import RouletteSelection, TournamentSelection
from hgp_lib.utils.metrics import fast_f1_score
from hgp_lib.utils.trial_details import extract_trial_details, store_trial_details
from hgp_lib.utils.visualization import (
    plot_all_runs_progression,
    plot_epoch_progression,
    plot_hierarchical_progression,
)

# Use non-interactive backend for matplotlib (required for saving plots without display)
matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> Tuple[pd.DataFrame, ndarray]:
    """
    Load and preprocess data from HDF file for benchmarking.

    Loads the data, identifies the target column, and binarizes features
    using StandardBinarizer. Supports PaySim format (isFraud column) and
    generic format (target column).

    Args:
        data_path: Path to HDF file containing data.
        num_bins: Number of bins for binarization. Default: 5.

    Returns:
        Tuple of (data, labels) as numpy arrays, binarized and ready for GP.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If target column cannot be identified.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from {data_path}...")

    df: pd.DataFrame = pd.read_hdf(data_path)

    # Detect target column - PaySim uses isFraud, others use target
    if "isFraud" in df.columns:
        target_column = "isFraud"
    elif "target" in df.columns:
        target_column = "target"
    else:
        raise RuntimeError(f"Unknown target column. Available: {df.columns.tolist()}")

    labels = df[target_column].to_numpy(dtype=bool, copy=True)
    data = df.drop([target_column], axis=1)

    del df
    return data, labels


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Tuning for Boolean GP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to HDF data file"
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of optimization trials"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="gp_hypertuning",
        help="Name for the Optuna study",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="./optuna_study.db",
        help="Path for SQLite database",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="./artifacts",
        help="Directory for storing trial artifacts (plots)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of parallel jobs (-1 = all CPUs)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress bars for runs/folds/epochs",
    )
    return parser


def suggest_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest all hyperparameters (base + hierarchical) in one function."""
    params = {}

    params["num_bins"] = trial.suggest_int("num_bins", 4, 6)

    # Base GP parameters
    params["population_size"] = trial.suggest_int("population_size", 50, 200, step=25)
    params["mutation_probability"] = trial.suggest_float(
        "mutation_probability", 0.001, 0.3, step=0.001
    )
    params["crossover_rate"] = trial.suggest_float(
        "crossover_rate", 0.1, 0.95, step=0.05
    )
    params["num_epochs"] = trial.suggest_int("num_epochs", 100, 5000, step=100)

    params["selection_type"] = trial.suggest_categorical(
        "selection_type", ["roulette", "tournament"]
    )
    if params["selection_type"] == "tournament":
        params["tournament_size"] = trial.suggest_int("tournament_size", 5, 30)
        params["selection_p"] = trial.suggest_float("selection_p", 0.25, 0.9, step=0.05)

    params["regeneration"] = trial.suggest_categorical("regeneration", [True, False])
    if params["regeneration"]:
        min_regen = 50
        max_regen = params["num_epochs"] // 2
        max_regen -= max_regen % min_regen
        params["regeneration_patience"] = trial.suggest_int(
            "regeneration_patience", min_regen, max_regen, step=min_regen
        )

    # Hierarchical GP parameters (num_child_populations=0 means no hierarchy)
    params["num_child_populations"] = trial.suggest_int("num_child_populations", 0, 10)

    if params["num_child_populations"] > 0:
        params["max_depth"] = 1
        params["top_k_transfer"] = trial.suggest_int("top_k_transfer", 5, 50, step=5)
        params["feedback_type"] = trial.suggest_categorical(
            "feedback_type", ["additive", "multiplicative"]
        )
        params["feedback_strength"] = trial.suggest_float(
            "feedback_strength", 0.0, 0.2, step=0.01
        )

        params["sampling_strategy_type"] = trial.suggest_categorical(
            "sampling_strategy_type", ["feature", "instance", "combined"]
        )
        params["use_replace"] = trial.suggest_categorical("use_replace", [True, False])

        if not params["use_replace"]:
            max_fraction = 1.0 / params["num_child_populations"]
        else:
            max_fraction = 1.0

        if params["sampling_strategy_type"] in ("feature", "combined"):
            params["feature_fraction"] = trial.suggest_float(
                "feature_fraction", 0.1, max_fraction, step=0.01
            )
        if params["sampling_strategy_type"] in ("instance", "combined"):
            params["sample_fraction"] = trial.suggest_float(
                "sample_fraction", 0.1, max_fraction, step=0.01
            )

    return params


def build_config(
    params: Dict[str, Any],
    data: pd.DataFrame,
    labels: ndarray,
    score_fn: Callable,
    n_jobs: int,
    verbose: bool = False,
) -> BenchmarkerConfig:
    """Build BenchmarkerConfig from suggested hyperparameters."""
    # Selection strategy
    if params["selection_type"] == "tournament":
        selection = TournamentSelection(
            tournament_size=params["tournament_size"], selection_p=params["selection_p"]
        )
    else:
        selection = RouletteSelection()

    # Population generator
    population_size = params["population_size"]

    # Mutation and crossover
    mutation_p = params["mutation_probability"]
    crossover_executor = CrossoverExecutor(crossover_p=params["crossover_rate"])

    # Sampling strategy for hierarchical GP
    sampling_strategy = None
    if params.get("num_child_populations", 0) > 0:
        use_replace = params.get("use_replace", False)
        strategy_type = params.get("sampling_strategy_type", "feature")
        if strategy_type == "feature":
            sampling_strategy = FeatureSamplingStrategy(
                feature_fraction=params.get("feature_fraction", 1.0),
                replace=use_replace,
            )
        elif strategy_type == "instance":
            sampling_strategy = InstanceSamplingStrategy(
                sample_fraction=params.get("sample_fraction", 1.0),
                replace=use_replace,
            )
        else:
            sampling_strategy = CombinedSamplingStrategy(
                feature_fraction=params.get("feature_fraction", 1.0),
                sample_fraction=params.get("sample_fraction", 1.0),
                replace=use_replace,
            )

    # Build configs
    gp_config = BooleanGPConfig(
        score_fn=score_fn,
        selection=selection,
        population_factory=PopulationGeneratorFactory(population_size=population_size),
        mutation_factory=MutationExecutorFactory(mutation_p=mutation_p),
        crossover_executor=crossover_executor,
        regeneration=params.get("regeneration", False),
        regeneration_patience=params.get("regeneration_patience", 100),
        num_child_populations=params.get("num_child_populations", 0),
        max_depth=params.get("max_depth", 0),
        sampling_strategy=sampling_strategy,
        top_k_transfer=params.get("top_k_transfer", 10),
        feedback_type=params.get("feedback_type", "multiplicative"),
        feedback_strength=params.get("feedback_strength", 0.1),
    )

    trainer_config = TrainerConfig(
        gp_config=gp_config, num_epochs=params["num_epochs"], progress_bar=verbose
    )

    binarizer = StandardBinarizer(num_bins=params["num_bins"])

    return BenchmarkerConfig(
        data=data,
        labels=labels,
        trainer_config=trainer_config,
        binarizer=binarizer,
        num_runs=5,
        n_folds=3,
        n_jobs=n_jobs,
        show_run_progress=verbose,
        show_fold_progress=verbose,
        show_epoch_progress=verbose,
    )


def create_objective(
    data: pd.DataFrame,
    labels: ndarray,
    n_jobs: int,
    artifact_store: FileSystemArtifactStore,
    verbose: bool = False,
) -> Callable[[optuna.Trial], float]:
    """Create the Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_hyperparameters(trial)

        try:
            config = build_config(params, data, labels, fast_f1_score, n_jobs, verbose)
            result = GPBenchmarker(config).fit()

            # Extract and store detailed trial information
            details = extract_trial_details(result)
            store_trial_details(trial, details)

            # Generate and save visualization artifacts
            _save_trial_artifacts(trial, result, details, artifact_store)

            logger.info(
                f"Trial {trial.number}: val={result.mean_best_val_score:.4f}, "
                f"test={result.mean_test_score:.4f}, hierarchical={params['num_child_populations'] > 0}"
            )

            return result.mean_best_val_score

        except Exception:
            logger.error(f"Trial {trial.number} failed: {traceback.format_exc()}")
            raise optuna.TrialPruned()

    return objective


def _save_trial_artifacts(
    trial: optuna.Trial,
    result,
    details,
    artifact_store: FileSystemArtifactStore,
) -> None:
    """
    Generate and save visualization artifacts for a trial.

    Handles edge cases gracefully:
    - Skips artifact generation if training history is not available
    - Skips hierarchical plot for non-hierarchical trials (is_hierarchical=False)
    - Logs warnings instead of failing on artifact upload errors

    Args:
        trial: Optuna trial object.
        result: BenchmarkResult containing run metrics.
        details: TrialDetails with extracted trial information.
        artifact_store: FileSystemArtifactStore for saving artifacts.
    """
    import matplotlib.pyplot as plt

    # Validate best_run_idx is within bounds
    if details.best_run_idx >= len(result.run_metrics):
        logger.warning(
            f"Trial {trial.number}: best_run_idx ({details.best_run_idx}) "
            f"out of bounds, skipping artifacts"
        )
        return

    best_run = result.run_metrics[details.best_run_idx]

    # Create a temporary directory for saving plots
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Epoch progression plot (best run)
        if best_run.train_history is not None:
            try:
                fig = plot_epoch_progression(
                    best_run.train_history,
                    best_run.val_history,
                    title=f"Trial {trial.number}: F1 Score Progression (Best Run)",
                )
                epoch_path = os.path.join(tmpdir, "epoch_progression.png")
                fig.savefig(epoch_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                upload_artifact(trial, epoch_path, artifact_store)
            except Exception as e:
                logger.error(
                    f"Trial {trial.number}: Failed to save epoch progression plot: {e}"
                )

        # 2. All runs overlay plot
        runs_with_history = [
            rm for rm in result.run_metrics if rm.val_history is not None
        ]
        if runs_with_history:
            try:
                fig = plot_all_runs_progression(
                    result.run_metrics,
                    details.best_run_idx,
                )
                all_runs_path = os.path.join(tmpdir, "all_runs_progression.png")
                fig.savefig(all_runs_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                upload_artifact(trial, all_runs_path, artifact_store)
            except Exception as e:
                logger.warning(
                    f"Trial {trial.number}: Failed to save all runs plot: {e}"
                )

        # 3. Hierarchical plot (only for hierarchical trials)
        # Skip for non-hierarchical trials (is_hierarchical=False or num_children=0)
        if not details.is_hierarchical or details.num_children == 0:
            pass
        elif best_run.train_history is not None:
            # Verify children scores are actually available before plotting
            has_children_scores = any(
                epoch.children_best_scores is not None
                for epoch in best_run.train_history.epochs
            )
            if has_children_scores:
                try:
                    fig = plot_hierarchical_progression(
                        best_run.train_history,
                        details.num_children,
                    )
                    hier_path = os.path.join(tmpdir, "hierarchical_progression.png")
                    fig.savefig(hier_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    upload_artifact(trial, hier_path, artifact_store)
                except Exception as e:
                    logger.warning(
                        f"Trial {trial.number}: Failed to save hierarchical plot: {e}"
                    )
            else:
                logger.warning(
                    f"Trial {trial.number}: Hierarchical trial but no children "
                    "scores available, skipping hierarchical plot"
                )


def main() -> None:
    """Main entry point for the hyperparameter tuning script."""
    parser = create_argument_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    logger.info("Loading data...")
    data, labels = load_data(args.data_path)

    # Initialize artifact store
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_store = FileSystemArtifactStore(base_path=str(artifact_dir))
    logger.info(f"Artifact store initialized at: {artifact_dir}")

    storage = f"sqlite:///{args.storage_path}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    # Log info about existing trials (handle backward compatibility)
    existing_trials = study.trials
    if existing_trials:
        trials_with_artifacts = sum(
            1
            for t in existing_trials
            if t.user_attrs.get("is_hierarchical") is not None
        )
        logger.info(
            f"Loaded existing study with {len(existing_trials)} trials "
            f"({trials_with_artifacts} with detailed attributes)"
        )
        if trials_with_artifacts < len(existing_trials):
            logger.info(
                "Note: Some older trials may not have detailed attributes or artifacts. "
                "These will be displayed with limited information in the dashboard."
            )

    objective = create_objective(
        data, labels, args.n_jobs, artifact_store, args.verbose
    )

    logger.info(f"Starting optimization with {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    best = study.best_trial
    print(f"\nBest validation score: {best.value:.4f}")

    # Safely access user attributes (handle old trials without attributes)
    best_test_score = best.user_attrs.get("best_run_test_score")
    is_hierarchical = best.user_attrs.get("is_hierarchical")

    if best_test_score is not None:
        print(f"Best test score: {best_test_score:.4f}")
    else:
        print("Best test score: N/A (trial from older version)")

    if is_hierarchical is not None:
        print(f"Is hierarchical: {is_hierarchical}")
    else:
        print("Is hierarchical: N/A (trial from older version)")

    print("\nBest hyperparameters:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    print(f"\nResults saved to: {args.storage_path}")
    print(f"Artifacts saved to: {args.artifact_dir}")
    print("\n" + "-" * 60)
    print("To view results with artifacts in Optuna Dashboard:")
    print(f"  optuna-dashboard {storage} --artifact-dir {args.artifact_dir}")
    print("-" * 60)


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/optuna_hypertuning.py --data-path data/PaySim.hdf --n-trials 100 --study-name 3_tests_3_val_folds --verbose --artifact-dir ./artifacts
# optuna-dashboard sqlite:///optuna_study.db --artifact-dir ./artifacts
