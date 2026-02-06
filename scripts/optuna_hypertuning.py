#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning Script for Boolean GP and Hierarchical GP.

This script uses Optuna to optimize hyperparameters for the BooleanGP algorithm.
It integrates with GPBenchmarker for statistically robust evaluation (30 runs, k-fold CV).
Results are stored in SQLite and viewable via Optuna Dashboard.

Usage:
    python scripts/optuna_hypertuning.py --data-path data/PaySim.hdf --n-trials 100

View results:
    optuna-dashboard sqlite:///optuna_study.db
"""

import argparse
import logging
from typing import Any, Callable, Dict

import numpy as np
import optuna
from numpy import ndarray

from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.mutations import create_mutation_executor
from hgp_lib.populations import (
    CombinedSamplingStrategy,
    FeatureSamplingStrategy,
    InstanceSamplingStrategy,
    PopulationGenerator,
    RandomStrategy,
)
from hgp_lib.selections import RouletteSelection, TournamentSelection
from hgp_lib.utils.benchmarking import load_data
from hgp_lib.utils.metrics import fast_f1_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    # Base GP parameters
    params["population_size"] = trial.suggest_int("population_size", 50, 250, step=25)
    params["mutation_probability"] = trial.suggest_float(
        "mutation_probability", 0.001, 0.7
    )
    params["crossover_rate"] = trial.suggest_float("crossover_rate", 0.1, 0.9)
    params["num_epochs"] = trial.suggest_int("num_epochs", 100, 10000, step=100)

    params["selection_type"] = trial.suggest_categorical(
        "selection_type", ["roulette", "tournament"]
    )
    if params["selection_type"] == "tournament":
        params["tournament_size"] = trial.suggest_int("tournament_size", 2, 20)
        params["selection_p"] = trial.suggest_float("selection_p", 0.1, 0.9)

    params["regeneration"] = trial.suggest_categorical("regeneration", [True, False])
    if params["regeneration"]:
        params["regeneration_patience"] = trial.suggest_int(
            "regeneration_patience", 50, 500
        )

    # Hierarchical GP parameters (num_child_populations=0 means no hierarchy)
    params["num_child_populations"] = trial.suggest_int("num_child_populations", 0, 5)

    if params["num_child_populations"] > 0:
        params["max_depth"] = trial.suggest_int("max_depth", 1, 3)
        params["top_k_transfer"] = trial.suggest_int("top_k_transfer", 5, 50)
        params["feedback_type"] = trial.suggest_categorical(
            "feedback_type", ["additive", "multiplicative"]
        )
        params["feedback_strength"] = trial.suggest_float(
            "feedback_strength", 0.0, 1.0
        )

        params["sampling_strategy_type"] = trial.suggest_categorical(
            "sampling_strategy_type", ["feature", "instance", "combined"]
        )

        if params["sampling_strategy_type"] in ("feature", "combined"):
            params["feature_fraction"] = trial.suggest_float(
                "feature_fraction", 0.1, 5.0
            )
        if params["sampling_strategy_type"] in ("instance", "combined"):
            params["instance_fraction"] = trial.suggest_float(
                "instance_fraction", 0.1, 5.0
            )

    return params


def build_config(
    params: Dict[str, Any],
    data: ndarray,
    labels: ndarray,
    score_fn: Callable,
    n_jobs: int,
    verbose: bool = False,
) -> BenchmarkerConfig:
    """Build BenchmarkerConfig from suggested hyperparameters."""
    num_features = data.shape[1]

    # Selection strategy
    if params["selection_type"] == "tournament":
        selection = TournamentSelection(
            tournament_size=params["tournament_size"], selection_p=params["selection_p"]
        )
    else:
        selection = RouletteSelection()

    # Population generator
    population_generator = PopulationGenerator(
        strategies=[RandomStrategy(num_literals=num_features)],
        population_size=params["population_size"],
    )

    # Mutation and crossover
    mutation_executor = create_mutation_executor(
        num_literals=num_features, mutation_p=params["mutation_probability"]
    )
    crossover_executor = CrossoverExecutor(crossover_p=params["crossover_rate"])

    # Sampling strategy for hierarchical GP
    sampling_strategy = None
    if params.get("num_child_populations", 0) > 0:
        strategy_type = params.get("sampling_strategy_type", "feature")
        if strategy_type == "feature":
            sampling_strategy = FeatureSamplingStrategy(
                feature_fraction=params.get("feature_fraction", 1.0)
            )
        elif strategy_type == "instance":
            sampling_strategy = InstanceSamplingStrategy(
                instance_fraction=params.get("instance_fraction", 1.0)
            )
        else:
            sampling_strategy = CombinedSamplingStrategy(
                feature_fraction=params.get("feature_fraction", 1.0),
                instance_fraction=params.get("instance_fraction", 1.0),
            )

    # Build configs
    gp_config = BooleanGPConfig(
        score_fn=score_fn,
        selection=selection,
        population_generator=population_generator,
        mutation_executor=mutation_executor,
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

    return BenchmarkerConfig(
        data=data,
        labels=labels,
        trainer_config=trainer_config,
        num_runs=3,
        n_folds=3,
        n_jobs=n_jobs,
        show_run_progress=verbose,
        show_fold_progress=verbose,
        show_epoch_progress=verbose,
    )


def create_objective(
    data: ndarray, labels: ndarray, n_jobs: int, verbose: bool = False
) -> Callable[[optuna.Trial], float]:
    """Create the Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_hyperparameters(trial)

        try:
            config = build_config(params, data, labels, fast_f1_score, n_jobs, verbose)
            result = GPBenchmarker(config).fit()

            # Store metrics as user attributes (visible in dashboard)
            trial.set_user_attr("mean_test_score", result.mean_test_score)
            trial.set_user_attr("std_test_score", result.std_test_score)
            trial.set_user_attr("std_val_score", result.std_best_val_score)
            trial.set_user_attr("is_hierarchical", params["num_child_populations"] > 0)

            logger.info(
                f"Trial {trial.number}: val={result.mean_best_val_score:.4f}, "
                f"test={result.mean_test_score:.4f}, hierarchical={params['num_child_populations'] > 0}"
            )

            return result.mean_best_val_score

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

    return objective


def main() -> None:
    """Main entry point for the hyperparameter tuning script."""
    parser = create_argument_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    logger.info("Loading data...")
    data, labels = load_data(args.data_path)

    storage = f"sqlite:///{args.storage_path}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    objective = create_objective(data, labels, args.n_jobs, args.verbose)

    logger.info(f"Starting optimization with {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    best = study.best_trial
    print(f"\nBest validation score: {best.value:.4f}")
    print(f"Best test score: {best.user_attrs.get('mean_test_score', 'N/A')}")
    print(f"Is hierarchical: {best.user_attrs.get('is_hierarchical', 'N/A')}")

    print("\nBest hyperparameters:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    print(f"\nResults saved to: {args.storage_path}")
    print(f"To view results: optuna-dashboard {storage}")


if __name__ == "__main__":
    main()

# python scripts/optuna_hypertuning.py --data-path data/PaySim.hdf --n-trials 100 --study-name 3_tests_3_val_folds --verbose