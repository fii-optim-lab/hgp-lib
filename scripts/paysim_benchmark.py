"""
PaySim Fraud Detection Benchmarking Script

Benchmarks Boolean GP on PaySim transaction data using multi-run evaluation
with k-fold cross-validation. Provides statistically robust performance
estimates with configurable hyperparameters.

==============================================================================
BENCHMARKING METHODOLOGY
==============================================================================

The benchmarker runs multiple independent experiments to estimate performance:

1. For each run (default 30):
   - Split data into train (80%) and test (20%) sets
   - Perform k-fold CV (default 5) on training set
   - Select best rule from fold with highest validation score
   - Evaluate on held-out test set

2. Report mean and std of test scores across all runs

This methodology provides:
- Robust performance estimates (not dependent on single random split)
- Confidence intervals for comparing algorithms
- Best rules from multiple random initializations

==============================================================================
USAGE EXAMPLES
==============================================================================

Basic benchmark with defaults:
    python scripts/paysim_benchmark.py

Quick benchmark (fewer runs/folds):
    python scripts/paysim_benchmark.py --num_runs 5 --n_folds 3 --num_epochs 500

Hierarchical GP benchmark:
    python scripts/paysim_benchmark.py --max_depth 1 --num_child_populations 3

Full benchmark with all CPUs:
    python scripts/paysim_benchmark.py --num_runs 30 --n_folds 5 --n_jobs -1

Compare configurations:
    python scripts/paysim_benchmark.py --max_depth 0 --num_epochs 1000
    python scripts/paysim_benchmark.py --max_depth 1 --num_epochs 1000

Requires preprocessed PaySim data in HDF format at data/PaySim.hdf
"""

import argparse
from functools import partial

import numpy as np
import pandas as pd

from hgp_lib import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.populations import (
    FeatureSamplingStrategy,
    PopulationGeneratorFactory,
)
from hgp_lib.preprocessing import StandardBinarizer
from hgp_lib.rules import Rule


# ==============================================================================
# SCORING FUNCTION
# ==============================================================================


def f1_score(y_pred, y_true, sample_weight=None):
    """
    F1 score implementation supporting sample weights for scorer optimization.

    When optimize_scorer=True in BooleanGPConfig, the scorer receives sample_weight
    that emphasizes minority class samples. This improves performance on imbalanced
    datasets like fraud detection.

    Args:
        y_pred: Boolean predictions array
        y_true: Boolean ground truth array
        sample_weight: Optional weights for each sample (used by scorer optimization)

    Returns:
        F1 score as float in [0, 1]
    """
    if sample_weight is None:
        y_pred_sum = y_pred.sum()
        y_true_sum = y_true.sum()
    else:
        y_pred_sum = np.dot(y_pred, sample_weight)
        y_true_sum = np.dot(y_true, sample_weight)

    if y_pred_sum == 0 or y_true_sum == 0:
        return 1.0 if y_pred_sum == y_true_sum == 0 else 0.0

    if sample_weight is None:
        tp = (y_pred & y_true).sum()
    else:
        tp = np.dot(y_pred & y_true, sample_weight)
    return 2 * tp / (y_pred_sum + y_true_sum)


# ==============================================================================
# DATA LOADING
# ==============================================================================


def load_paysim(hdf_path: str):
    print(f"Loading data from {hdf_path}...")
    df: pd.DataFrame = pd.read_hdf(hdf_path)

    # Detect target column (supports both original and preprocessed formats)
    if "isFraud" in df.columns:
        target_column = "isFraud"
    elif "target" in df.columns:
        target_column = "target"
    else:
        raise RuntimeError(f"Target column not found. Columns: {df.columns.tolist()}")

    labels = df[target_column].values.copy()
    data = df.drop([target_column], axis=1)
    del df

    print(f"Loaded {len(data)} samples, {len(data.columns)} features")
    print(f"Fraud rate: {labels.mean():.4f} ({labels.sum()} fraud cases)")

    return data, labels


def is_valid(rule: Rule, max_rule_size: int) -> bool:
    return len(rule) <= max_rule_size


def create_validity_checker(max_rule_size: int):
    """
    Create a function to check if a rule is valid.

    This is used during mutation and crossover to reject rules that are too large.
    Keeping rules small:
    - Improves interpretability
    - Prevents overfitting
    - Speeds up evaluation

    Args:
        max_rule_size: Maximum number of nodes in a rule tree

    Returns:
        Function that returns True if rule is valid
    """
    return partial(is_valid, max_rule_size=max_rule_size)


# ==============================================================================
# MAIN BENCHMARK FUNCTION
# ==============================================================================


def main(args: argparse.Namespace):
    """
    Main benchmarking function.

    Workflow:
    1. Load and binarize data
    2. Configure GP algorithm with specified parameters
    3. Configure benchmarker (runs, folds, parallelization)
    4. Run benchmark
    5. Print results with statistics
    """
    # Load data
    data, labels = load_paysim(args.data_path)

    # Create validity checker
    is_valid = create_validity_checker(args.max_rule_size)

    # Configure GP algorithm
    print("\n" + "=" * 60)
    print("GP CONFIGURATION")
    print("=" * 60)

    gp_config = BooleanGPConfig(
        score_fn=f1_score,
        check_valid=is_valid,
        population_factory=PopulationGeneratorFactory(
            population_size=args.population_size
        ),
        optimize_scorer=args.optimize_scorer,
        regeneration=args.regeneration,
        regeneration_patience=args.regeneration_patience,
    )

    # Apply hierarchical GP settings if max_depth > 0
    if args.max_depth > 0:
        print("Hierarchical GP enabled:")
        print(f"  max_depth: {args.max_depth}")
        print(f"  num_child_populations: {args.num_child_populations}")
        print(f"  feature_fraction: {args.feature_fraction}")

        gp_config.max_depth = args.max_depth
        gp_config.num_child_populations = args.num_child_populations
        gp_config.sampling_strategy = FeatureSamplingStrategy(
            feature_fraction=args.feature_fraction
        )
    else:
        print("Standard GP (no hierarchy)")

    print(f"\nPopulation size: {args.population_size}")
    print(f"Max rule size: {args.max_rule_size}")
    print(f"Optimize scorer: {args.optimize_scorer}")
    print(f"Regeneration: {args.regeneration} (patience={args.regeneration_patience})")

    # Configure trainer
    trainer_config = TrainerConfig(
        gp_config=gp_config,
        num_epochs=args.num_epochs,
        val_every=args.val_every,
        progress_bar=not args.no_progress,
    )

    print(f"\nTraining epochs: {args.num_epochs}")
    print(f"Validation every: {args.val_every} epochs")

    # Configure benchmarker
    print("\n" + "=" * 60)
    print("BENCHMARK CONFIGURATION")
    print("=" * 60)

    binarizer = StandardBinarizer(num_bins=args.num_bins)

    config = BenchmarkerConfig(
        data=data,
        labels=labels,
        trainer_config=trainer_config,
        binarizer=binarizer,
        num_runs=args.num_runs,
        n_folds=args.n_folds,
        test_size=args.test_size,
        n_jobs=args.n_jobs,
        base_seed=args.base_seed,
        show_run_progress=not args.no_progress,
        show_fold_progress=not args.no_progress,
        show_epoch_progress=not args.no_progress,
    )

    print(f"Number of runs: {config.num_runs}")
    print(f"Folds per run: {config.n_folds}")
    print(f"Test size: {config.test_size}")
    print(f"Parallel jobs: {config.n_jobs} (-1 = all CPUs)")
    print(f"Base seed: {config.base_seed}")

    # Calculate total training iterations
    total_folds = config.num_runs * config.n_folds
    total_epochs = total_folds * args.num_epochs
    print("\nTotal training:")
    print(
        f"  {total_folds} fold trainings ({config.num_runs} runs × {config.n_folds} folds)"
    )
    print(f"  {total_epochs:,} total epochs")

    # Run benchmark
    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK")
    print("=" * 60)

    benchmarker = GPBenchmarker(config)
    result = benchmarker.fit()

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print("\nConfiguration summary:")
    print(
        f"  max_depth={args.max_depth}, epochs={args.num_epochs}, "
        f"pop_size={args.population_size}"
    )
    if args.max_depth > 0:
        print(
            f"  child_pops={args.num_child_populations}, "
            f"feature_frac={args.feature_fraction}"
        )

    print("\nPerformance:")
    print(
        f"  Test F1 Score:  {result.mean_test_score:.4f} ± {result.std_test_score:.4f}"
    )
    print(
        f"  Val F1 Score:   {result.mean_best_val_score:.4f} ± {result.std_best_val_score:.4f}"
    )

    print("\nPer-run test scores:")
    for i, score in enumerate(result.all_test_scores):
        print(f"  Run {i:2d}: {score:.4f}")

    # Statistics
    scores = np.array(result.all_test_scores)
    print("\nStatistics:")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Max:    {scores.max():.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    print(f"  IQR:    {np.percentile(scores, 75) - np.percentile(scores, 25):.4f}")

    # Show best rule from run with highest test score
    best_run_idx = np.argmax(result.all_test_scores)
    best_rule = result.all_best_rules[best_run_idx]
    print(f"\nBest rule (from run {best_run_idx}, score={scores[best_run_idx]:.4f}):")
    print(f"  {best_rule}")
    print("\nReadable form:")
    print(f"  {best_rule.to_str(result.feature_names_per_run[best_run_idx])}")

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Arguments are organized into groups:
    - Data: Input file path, binarization settings
    - Training: Epochs, validation frequency
    - GP Algorithm: Population size, rule constraints
    - Hierarchical GP: Depth, child populations, feature sampling
    - Optimization: Scorer optimization, regeneration
    - Benchmark: Runs, folds, parallelization
    - Output: Progress bar control
    """
    parser = argparse.ArgumentParser(
        description="Benchmark Boolean GP on PaySim fraud detection data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data_path",
        type=str,
        default="data/PaySim.hdf",
        help="Path to preprocessed PaySim HDF file",
    )
    data_group.add_argument(
        "--num_bins",
        type=int,
        default=5,
        help="Number of bins for feature binarization",
    )

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of training epochs per fold",
    )
    train_group.add_argument(
        "--val_every",
        type=int,
        default=10,
        help="Validate every N epochs",
    )

    # GP algorithm arguments
    gp_group = parser.add_argument_group("GP Algorithm")
    gp_group.add_argument(
        "--population_size",
        type=int,
        default=100,
        help="Size of the rule population",
    )
    gp_group.add_argument(
        "--max_rule_size",
        type=int,
        default=50,
        help="Maximum nodes in a rule tree",
    )

    # Hierarchical GP arguments
    hier_group = parser.add_argument_group("Hierarchical GP")
    hier_group.add_argument(
        "--max_depth",
        type=int,
        default=0,
        help="Maximum depth of population hierarchy (0 = standard GP)",
    )
    hier_group.add_argument(
        "--num_child_populations",
        type=int,
        default=3,
        help="Number of child populations per parent (when max_depth > 0)",
    )
    hier_group.add_argument(
        "--feature_fraction",
        type=float,
        default=0.33,
        help="Fraction of features for each child population (when max_depth > 0)",
    )

    # Optimization arguments
    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument(
        "--optimize_scorer",
        action="store_true",
        default=True,
        help="Enable scorer optimization for imbalanced data",
    )
    opt_group.add_argument(
        "--no_optimize_scorer",
        action="store_false",
        dest="optimize_scorer",
        help="Disable scorer optimization",
    )
    opt_group.add_argument(
        "--regeneration",
        action="store_true",
        default=True,
        help="Enable population regeneration on stagnation",
    )
    opt_group.add_argument(
        "--no_regeneration",
        action="store_false",
        dest="regeneration",
        help="Disable population regeneration",
    )
    opt_group.add_argument(
        "--regeneration_patience",
        type=int,
        default=200,
        help="Epochs without improvement before regeneration",
    )

    # Benchmark arguments
    bench_group = parser.add_argument_group("Benchmark")
    bench_group.add_argument(
        "--num_runs",
        type=int,
        default=30,
        help="Number of independent benchmark runs",
    )
    bench_group.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds per run",
    )
    bench_group.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data held out for testing",
    )
    bench_group.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 = all CPUs, 1 = sequential)",
    )
    bench_group.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Base random seed (run i uses seed base_seed + i)",
    )

    # Output arguments
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bars (cleaner output for logging)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
