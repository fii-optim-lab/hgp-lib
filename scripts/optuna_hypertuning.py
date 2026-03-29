"""
Usage:
    python scripts/optuna_hypertuning.py --data-path data/PaySim.hdf --n-trials 100

View results:
    optuna-dashboard sqlite:///optuna_study.db --artifact-dir ./artifacts
"""

import argparse
import logging
import os
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import optuna
import pandas as pd
from numpy import ndarray
from optuna.artifacts import FileSystemArtifactStore
from optuna.trial import TrialState

from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.crossover import CrossoverExecutorFactory
from visualization.optuna import store_trial_attributes, upload_trial_artifacts
from hgp_lib.mutations import MutationExecutorFactory
from hgp_lib.populations import (
    CombinedSamplingStrategy,
    FeatureSamplingStrategy,
    InstanceSamplingStrategy,
    PopulationGeneratorFactory,
)
from hgp_lib.preprocessing import StandardBinarizer, load_data
from hgp_lib.selections import RouletteSelection, TournamentSelection
from hgp_lib.utils.metrics import fast_f1_score
from hgp_lib.utils.validation import ComplexityCheck

from hypertuning import load_search_space
from preprocess.pmlb_preprocess import save_pmlb_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def suggest_hyperparameters(
    trial: optuna.Trial, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Suggest hyperparameters using ranges from ``config``.

    ``config`` maps parameter names to ``(args, kwargs)`` tuples as returned
    by ``load_search_space``.  Each ``trial.suggest_*`` call unpacks them
    directly.  Fallback defaults are provided so the function works even
    with an empty config.
    """
    warnings.simplefilter(action="ignore", category=UserWarning)

    def get(name, default_args=(), default_kw=None):
        a, k = config.get(name, (default_args, default_kw or {}))
        return a, {**(default_kw or {}), **k}

    params = {}

    a, k = get("max_complexity", (80,))
    if len(a) == 1:
        params["max_complexity"] = a[0]
    else:
        params["max_complexity"] = trial.suggest_int("max_complexity", *a, **k)

    a, k = get("num_bins", (2, 10))
    params["num_bins"] = trial.suggest_int("num_bins", *a, **k)

    # Base GP parameters
    a, k = get("population_size", (50, 150), {"step": 25})
    params["population_size"] = trial.suggest_int("population_size", *a, **k)

    a, k = get("mutation_probability", (0.001, 0.25), {"step": 0.001})
    params["mutation_probability"] = trial.suggest_float(
        "mutation_probability", *a, **k
    )

    a, k = get("crossover_rate", (0.1, 0.95), {"step": 0.05})
    params["crossover_rate"] = trial.suggest_float("crossover_rate", *a, **k)

    a, k = get("crossover_operator_p", (0.0, 1.0), {"step": 0.05})
    params["crossover_operator_p"] = trial.suggest_float(
        "crossover_operator_p", *a, **k
    )

    a, k = get("mutation_operator_p", (0.0, 1.0), {"step": 0.05})
    params["mutation_operator_p"] = trial.suggest_float("mutation_operator_p", *a, **k)

    a, k = get("num_epochs", (500, 3000), {"step": 100})
    params["num_epochs"] = trial.suggest_int("num_epochs", *a, **k)

    # Selection
    params["selection_type"] = get("selection_type", ("tournament",))[0][0]
    if params["selection_type"] == "tournament":
        a, k = get("tournament_size", (5, 30))
        params["tournament_size"] = trial.suggest_int("tournament_size", *a, **k)
        a, k = get("selection_p", (0.2, 0.9), {"step": 0.05})
        params["selection_p"] = trial.suggest_float("selection_p", *a, **k)

    # Regeneration
    a, k = get("regeneration", ([True, False],))
    params["regeneration"] = trial.suggest_categorical("regeneration", *a, **k)
    if params["regeneration"]:
        regen_step = get("regeneration_patience_step", (50,))[0][0]
        max_regen = params["num_epochs"] // 3
        max_regen -= max_regen % regen_step
        params["regeneration_patience"] = trial.suggest_int(
            "regeneration_patience", regen_step, max_regen, step=regen_step
        )

    # Complexity regularization
    a, k = get("use_complexity_penalty", ([True, False],))
    params["use_complexity_penalty"] = trial.suggest_categorical(
        "use_complexity_penalty", *a, **k
    )
    if params["use_complexity_penalty"]:
        a, k = get("complexity_penalty", (0.0, 0.1), {"step": 0.0001})
        params["complexity_penalty"] = trial.suggest_float(
            "complexity_penalty", *a, **k
        )

    # Hierarchical GP
    a, k = get("max_depth", (0, 2), {"step": 1})
    params["max_depth"] = trial.suggest_int("max_depth", *a, **k)

    if params["max_depth"] > 0:
        if params["max_depth"] >= 2:
            params["num_child_populations"] = 2
        else:
            a, k = get("num_child_populations", (2, 5))
            params["num_child_populations"] = trial.suggest_int(
                "num_child_populations", *a, **k
            )

        tk_args, tk_kw = get("top_k_transfer", (10, 100), {"step": 5})
        tk_high = min(tk_args[1], params["population_size"] - 1)
        params["top_k_transfer"] = trial.suggest_int(
            "top_k_transfer", tk_args[0], tk_high, **tk_kw
        )

        a, k = get("feedback_type", (["additive", "multiplicative"],))
        params["feedback_type"] = trial.suggest_categorical("feedback_type", *a, **k)

        a, k = get("feedback_strength", (0.0, 0.2), {"step": 0.01})
        params["feedback_strength"] = trial.suggest_float("feedback_strength", *a, **k)

        a, k = get("sampling_strategy_type", (["feature", "instance", "combined"],))
        params["sampling_strategy_type"] = trial.suggest_categorical(
            "sampling_strategy_type", *a, **k
        )

        params["use_replace"] = get("use_replace", (True,))[0][0]

        if not params["use_replace"]:
            max_fraction = 1.0 / (
                params["num_child_populations"] ** params["max_depth"]
            )
        else:
            max_fraction = 1.0

        low_frac = 0.1 * params["max_depth"]

        if params["sampling_strategy_type"] in ("feature", "combined"):
            ff_args, ff_kw = get(
                "feature_fraction", (low_frac, max_fraction), {"step": 0.01}
            )
            ff_high = min(ff_args[1], max_fraction)
            ff_low = ff_args[0]
            if ff_high - ff_low < (ff_kw.get("step", 0.01)):
                ff_kw.pop("step", None)
            params["feature_fraction"] = trial.suggest_float(
                "feature_fraction", ff_low, ff_high, **ff_kw
            )

        if params["sampling_strategy_type"] in ("instance", "combined"):
            sf_args, sf_kw = get(
                "sample_fraction", (low_frac, max_fraction), {"step": 0.01}
            )
            sf_high = min(sf_args[1], max_fraction)
            sf_low = sf_args[0]
            if sf_high - sf_low < (sf_kw.get("step", 0.01)):
                sf_kw.pop("step", None)
            params["sample_fraction"] = trial.suggest_float(
                "sample_fraction", sf_low, sf_high, **sf_kw
            )

    return params


def build_config(
    params: Dict[str, Any],
    data: pd.DataFrame,
    labels: ndarray,
    score_fn: Callable,
    n_jobs: int,
    n_runs: int,
    n_folds: int,
    verbose: bool = False,
) -> BenchmarkerConfig:
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
    check_valid = ComplexityCheck(params["max_complexity"])

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
        mutation_factory=MutationExecutorFactory(
            mutation_p=mutation_p, operator_p=params.get("mutation_operator_p", 0.5)
        ),
        crossover_factory=CrossoverExecutorFactory(
            crossover_p=params["crossover_rate"],
            operator_p=params.get("crossover_operator_p", 0.9),
        ),
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
        # TODO: we should make these configurable
        num_runs=n_runs,
        n_folds=n_folds,
        n_jobs=n_jobs,
        show_run_progress=verbose,
        show_fold_progress=verbose,
        show_epoch_progress=verbose,
    )


def create_objective(
    data: pd.DataFrame,
    labels: ndarray,
    n_jobs: int,
    n_runs: int,
    n_folds: int,
    artifact_store: FileSystemArtifactStore,
    hp_config: Dict[str, Any],
    verbose: bool = False,
) -> Callable[[optuna.Trial], float]:
    """Create the Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_hyperparameters(trial, hp_config)

        try:
            config = build_config(
                params, data, labels, fast_f1_score, n_jobs, n_runs, n_folds, verbose
            )
            result = GPBenchmarker(config).fit()
        except Exception:
            logger.error(f"Trial {trial.number} failed: {traceback.format_exc()}")
            raise optuna.TrialPruned()

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

    return objective


def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    logger.info("Loading data...")
    if not os.path.isfile(args.data_path):
        data_path = Path(args.data_path)
        try:
            save_pmlb_data(data_path.stem, str(data_path.parent))
        except FileNotFoundError:
            logging.error(f"Dataset {data_path.stem} not found in PMLB")
            traceback.print_exc()
            return
    data, labels = load_data(args.data_path)

    # Initialize artifact store
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_store = FileSystemArtifactStore(base_path=str(artifact_dir))
    logger.info(f"Artifact store initialized at: {artifact_dir}")

    # Load hyperparameter search space config
    hp_config = {}
    if args.hp_config is not None:
        hp_config = load_search_space(args.hp_config)
        logger.info(f"Loaded search space from: {args.hp_config}")

    storage = f"sqlite:///{args.storage_path}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    # Log info about existing trials
    existing_trials = study.trials
    completed_trials = 0
    if existing_trials:
        completed_trials = sum(
            [trial.state == TrialState.COMPLETE for trial in existing_trials]
        )
        # TODO: Debug this and add remaining trials
        logger.info(
            f"Loaded existing study with {len(existing_trials)} trials "
            f"({completed_trials} completed trials)"
        )

    max_n_trials = args.max_n_trials
    if completed_trials >= max_n_trials:
        logger.info(
            f"Study {args.study_name} with {len(existing_trials)} trials already completed."
        )
        return

    n_trials = min(args.n_trials, max_n_trials - completed_trials)

    objective = create_objective(
        data,
        labels,
        args.n_jobs,
        args.n_runs,
        args.n_folds,
        artifact_store,
        hp_config,
        args.verbose,
    )

    logger.info(f"Starting optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

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


def parse_args() -> argparse.Namespace:
    """Create and configure the argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Tuning for Boolean GP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to HDF data file"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials for this run",
    )
    parser.add_argument(
        "--max-n-trials",
        type=int,
        default=100,
        help="Maximum number of optimization trials for this study",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        required=True,
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
        "--n-runs",
        type=int,
        default=10,
        help="Number of Monte-Carlo runs for GPBenchmarker",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for k-fold cross-validation in GPBenchmarker",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress bars for runs/folds/epochs",
    )
    parser.add_argument(
        "--hp-config",
        type=str,
        default=None,
        help="Path to YAML hyperparameter search space config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

# Example usage:
# python scripts/optuna_hypertuning.py --data-path data/PaySim.hdf
#                                      --n-trials 100
#                                      --study-name PaySim
#                                      --verbose --artifact-dir ./artifacts
# python scripts/optuna_hypertuning.py --data-path data/breast_cancer.hdf
#                                      --n-trials 100
#                                      --study-name pmlb_breast_cancer
#                                      --verbose --artifact-dir ./artifacts
# optuna-dashboard sqlite:///optuna_study.db --artifact-dir ./artifacts


# source /Users/user/miniforge3/bin/activate
# conda activate 312
