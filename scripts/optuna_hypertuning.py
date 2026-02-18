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
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import matplotlib
import numpy as np
import optuna
import pandas as pd
from numpy import ndarray
from optuna.artifacts import FileSystemArtifactStore

from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.crossover import CrossoverExecutor
from visualization.optuna import store_trial_attributes, upload_trial_artifacts
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
from hgp_lib.utils.validation import complexity_check

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> Tuple[pd.DataFrame, ndarray]:
    """
    Load and preprocess data from HDF file for benchmarking.

    Args:
        data_path: Path to HDF file containing data.

    Returns:
        Tuple of (data, labels).

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

    params["num_bins"] = trial.suggest_int("num_bins", 4, 8)

    # Base GP parameters
    params["population_size"] = trial.suggest_int("population_size", 50, 200, step=25)
    params["mutation_probability"] = trial.suggest_float(
        "mutation_probability", 0.001, 0.25, step=0.001
    )
    params["crossover_rate"] = trial.suggest_float(
        "crossover_rate", 0.1, 0.95, step=0.05
    )
    params["crossover_operator_p"] = trial.suggest_float(
        "crossover_operator_p", 0.0, 1.0, step=0.05
    )
    params["mutation_operator_p"] = trial.suggest_float(
        "mutation_operator_p", 0.0, 1.0, step=0.05
    )
    params["num_epochs"] = trial.suggest_int("num_epochs", 500, 3000, step=100)

    params["selection_type"] = "tournament"
    if params["selection_type"] == "tournament":
        params["tournament_size"] = trial.suggest_int("tournament_size", 5, 30)
        params["selection_p"] = trial.suggest_float("selection_p", 0.2, 0.9, step=0.05)

    params["regeneration"] = trial.suggest_categorical("regeneration", [True, False])
    if params["regeneration"]:
        min_regen = 50
        max_regen = params["num_epochs"] // 3
        max_regen -= max_regen % min_regen
        params["regeneration_patience"] = trial.suggest_int(
            "regeneration_patience", min_regen, max_regen, step=min_regen
        )

    # Complexity regularization
    params["use_complexity_penalty"] = trial.suggest_categorical(
        "use_complexity_penalty", [True, False]
    )
    if params["use_complexity_penalty"]:
        params["complexity_penalty"] = trial.suggest_float(
            "complexity_penalty", 0.0, 0.1, step=0.0001
        )

    # Hierarchical GP parameters (num_child_populations=0 means no hierarchy)
    params["max_depth"] = trial.suggest_int("max_depth", 0, 2, step=1)
    if params["max_depth"] > 0:
        if params["max_depth"] == 3:
            params["num_child_populations"] = 2
        elif params["max_depth"] == 2:
            params["num_child_populations"] = trial.suggest_int(
                "num_child_populations", 2, 3
            )
        else:  # 1
            params["num_child_populations"] = trial.suggest_int(
                "num_child_populations", 2, 8
            )

        params["top_k_transfer"] = trial.suggest_int(
            "top_k_transfer", 10, min(100, params["population_size"]), step=5
        )
        params["feedback_type"] = trial.suggest_categorical(
            "feedback_type", ["additive", "multiplicative"]
        )
        params["feedback_strength"] = trial.suggest_float(
            "feedback_strength", 0.0, 0.2, step=0.01
        )

        params["sampling_strategy_type"] = trial.suggest_categorical(
            "sampling_strategy_type", ["feature", "instance", "combined"]
        )
        # params["use_replace"] = trial.suggest_categorical("use_replace", [True, False])
        params["use_replace"] = True

        if not params["use_replace"]:
            max_fraction = 1.0 / (
                params["num_child_populations"] ** params["max_depth"]
            )
        else:
            max_fraction = 1.0

        low = 0.1 * params["max_depth"]

        if params["sampling_strategy_type"] in ("feature", "combined"):
            if max_fraction - 0.1 >= 0.01:
                params["feature_fraction"] = trial.suggest_float(
                    "feature_fraction", low, max_fraction, step=0.01
                )
            else:
                params["feature_fraction"] = trial.suggest_float(
                    "feature_fraction", low, max_fraction
                )
        if params["sampling_strategy_type"] in ("instance", "combined"):
            if max_fraction - 0.1 >= 0.01:
                params["sample_fraction"] = trial.suggest_float(
                    "sample_fraction", low, max_fraction, step=0.01
                )
            else:
                params["sample_fraction"] = trial.suggest_float(
                    "sample_fraction", low, max_fraction
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
    check_valid = complexity_check()
    crossover_executor = CrossoverExecutor(
        crossover_p=params["crossover_rate"],
        check_valid=check_valid,
        operator_p=params.get("crossover_operator_p", 0.9),
    )

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
        mutation_factory=MutationExecutorFactory(mutation_p=mutation_p, operator_p=params.get("mutation_operator_p", 0.9)),
        crossover_executor=crossover_executor,
        check_valid=check_valid,
        regeneration=params.get("regeneration", False),
        regeneration_patience=params.get("regeneration_patience", 100),
        complexity_penalty=params.get("complexity_penalty", 0.0),
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
        num_runs=10,
        n_folds=5,
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

            # Store trial attributes using new metrics system
            store_trial_attributes(trial, result)

            # Upload visualization artifacts
            upload_trial_artifacts(
                trial,
                result,
                artifact_store,
                top_k_transfer=params.get("top_k_transfer", 10),
            )

            # Get mean validation score for optimization
            mean_val_score = float(np.mean([run.mean_val_score for run in result.runs]))

            mean_test_score = float(np.mean(result.test_scores))

            logger.info(
                f"Trial {trial.number}: val={mean_val_score:.4f}, "
                f"test={mean_test_score:.4f}, "
                f"hierarchical={params['max_depth'] > 0}"
            )

            return mean_val_score

        except Exception:
            logger.error(f"Trial {trial.number} failed: {traceback.format_exc()}")
            raise optuna.TrialPruned()

    return objective


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
    )

    # Log info about existing trials
    existing_trials = study.trials
    if existing_trials:
        trials_with_attrs = sum(
            1
            for t in existing_trials
            if t.user_attrs.get("06_hierarchy_is_hierarchical") is not None
        )
        logger.info(
            f"Loaded existing study with {len(existing_trials)} trials "
            f"({trials_with_attrs} with new attributes format)"
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

    # Access user attributes with new naming
    best_test_score = best.user_attrs.get("04_best_test_score")
    is_hierarchical = best.user_attrs.get("06_hierarchy_is_hierarchical")

    if best_test_score is not None:
        print(f"Best test score: {best_test_score:.4f}")

    if is_hierarchical is not None:
        print(f"Is hierarchical: {is_hierarchical}")

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


# source /Users/user/miniforge3/bin/activate
# conda activate 312
