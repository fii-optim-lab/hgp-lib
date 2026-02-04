"""
PaySim Fraud Detection Training Script with Profiling Support

This script trains a Boolean GP model on PaySim transaction data with comprehensive
profiling capabilities. It's designed to measure the performance impact of various
hyperparameters, especially tree depth and hierarchical population settings.

==============================================================================
PROFILING ARCHITECTURE
==============================================================================

The script uses `timed_decorator` to instrument key functions across the codebase.
Timing is collected with nested support, meaning we can see both:
- Total elapsed time (including time spent in child functions)
- Own time (excluding child function calls)

Key components profiled:
1. Rule evaluation (Literal, And, Or) - Core boolean operations
2. GP algorithm steps (_new_generation, _evaluate_population, _update_best)
3. Genetic operators (CrossoverExecutor, MutationExecutor)
4. Selection strategies (TournamentSelection, RouletteSelection)
5. Child population operations (for hierarchical GP)

==============================================================================
DEPTH PARAMETER IMPACT
==============================================================================

The `max_depth` parameter controls hierarchical GP:
- max_depth=0: Standard GP with single population (fastest)
- max_depth=1: One level of child populations
- max_depth=2+: Deeper hierarchies (exponentially more computation)

Each depth level multiplies computation by `num_child_populations`.
Use this script to measure the actual overhead vs accuracy tradeoff.

==============================================================================
USAGE EXAMPLES
==============================================================================

Basic profiling (standard GP):
    python scripts/paysim_trainer_profile.py --num_epochs 500

Profile hierarchical GP with depth 1:
    python scripts/paysim_trainer_profile.py --max_depth 1 --num_epochs 500

Compare depths with same epoch budget:
    python scripts/paysim_trainer_profile.py --max_depth 0 --num_epochs 1000
    python scripts/paysim_trainer_profile.py --max_depth 1 --num_epochs 1000
    python scripts/paysim_trainer_profile.py --max_depth 2 --num_epochs 1000

Adjust population and feature sampling:
    python scripts/paysim_trainer_profile.py --max_depth 1 \\
        --num_child_populations 5 --feature_fraction 0.5

Requires preprocessed PaySim data in HDF format at data/PaySim.hdf
"""

import random
from functools import partial
from prettytable import PrettyTable
import argparse
import gc
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from timed_decorator.builder import create_timed_decorator, get_timed_decorator

from hgp_lib import BooleanGPConfig, TrainerConfig
from hgp_lib.algorithms import BooleanGP
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.mutations import (
    AddLiteral,
    DeleteMutation,
    MutationExecutor,
    NegateMutation,
    PromoteLiteral,
    RemoveIntermediateOperator,
    ReplaceLiteral,
    ReplaceOperator,
)
from hgp_lib.populations import (
    FeatureSamplingStrategy,
    PopulationGenerator,
    RandomStrategy,
)
from hgp_lib.preprocessing import StandardBinarizer
from hgp_lib.rules import And, Literal, Or, Rule
from hgp_lib.selections import TournamentSelection
from hgp_lib.trainers import GPTrainer


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
# DATA PREPROCESSING
# ==============================================================================


def preprocess_paysim_data(hdf_path: str) -> Tuple:
    """
    Load and preprocess PaySim data for training.

    This function:
    1. Loads the HDF file containing transaction data
    2. Splits into train/val/test sets (64%/16%/20%)
    3. Binarizes features using StandardBinarizer

    The binarization is crucial for Boolean GP - it converts continuous features
    into boolean features that can be used as literals in the evolved rules.

    Args:
        hdf_path: Path to the preprocessed PaySim HDF file

    Returns:
        Tuple of (train_data, train_labels, val_data, val_labels,
                  test_data, test_labels, feature_names)
    """
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

    print(f"Loaded {len(data)} samples with {len(data.columns)} features")
    print(f"Fraud rate: {labels.mean():.4f} ({labels.sum()} fraud cases)")

    # Stratified splits to maintain fraud ratio across sets
    print("\nSplitting data...")
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    del data, labels

    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data,
        train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels,
    )

    print(f"Train: {len(train_data)} samples ({train_labels.sum()} fraud)")
    print(f"Val:   {len(val_data)} samples ({val_labels.sum()} fraud)")
    print(f"Test:  {len(test_data)} samples ({test_labels.sum()} fraud)")

    # Binarize features - converts continuous to boolean
    # num_bins=5 means each feature becomes 5 boolean features (one per quantile bin)
    print("\nBinarizing features...")
    binarizer = StandardBinarizer(num_bins=5)
    train_data_bin = binarizer.fit_transform(train_data, train_labels)

    # Create feature name mapping for interpretable rule output
    feature_names = {i: col for i, col in enumerate(train_data_bin.columns)}

    val_data_bin = binarizer.transform(val_data)
    test_data_bin = binarizer.transform(test_data)

    print(f"Binarized features: {train_data_bin.shape[1]} (from {train_data.shape[1]})")

    # Convert to numpy arrays for efficient computation
    train_data_bin = train_data_bin.values
    val_data_bin = val_data_bin.values
    test_data_bin = test_data_bin.values

    del train_data, val_data, test_data
    gc.collect()

    return (
        train_data_bin,
        train_labels,
        val_data_bin,
        val_labels,
        test_data_bin,
        test_labels,
        feature_names,
    )


# ==============================================================================
# PROFILING SETUP
# ==============================================================================


def setup_timing() -> Dict:
    """
    Initialize the timing measurement system.

    Creates a timed decorator that:
    - Supports nested timing (own_time vs elapsed_time)
    - Disables GC during timing for accurate measurements
    - Collects results in a dictionary for later analysis

    Returns:
        Dictionary that will be populated with timing measurements.
        Format: {func_name: (call_count, total_elapsed_ns, own_time_ns)}
    """
    measurements = {}
    create_timed_decorator(
        "GPTimer",
        nested=True,  # Track both elapsed and own time
        collect_gc=False,  # Don't trigger GC between measurements
        disable_gc=True,  # Disable GC during timed sections
        stdout=False,  # Don't print individual timings
        out=measurements,  # Collect results here
    )
    return measurements


def apply_timing_decorators() -> None:
    """
    Apply timing decorators to key functions throughout the codebase.

    This instruments the following components:

    1. RULE EVALUATION (core operations, called millions of times)
       - Literal.evaluate: Single boolean feature lookup
       - And.evaluate: Conjunction of child rules
       - Or.evaluate: Disjunction of child rules

    2. GP ALGORITHM (main training loop)
       - BooleanGP.step: One generation (selection + crossover + mutation + eval)
       - BooleanGP._new_generation: Create offspring via genetic operators
       - BooleanGP._evaluate_population: Score all rules against data
       - BooleanGP._update_best: Track best rule found so far
       - BooleanGP.validate_population: Evaluate on validation set

    3. GENETIC OPERATORS
       - CrossoverExecutor.apply: Combine pairs of rules
       - MutationExecutor.apply: Randomly modify rules

    4. SELECTION STRATEGIES
       - TournamentSelection.select: Tournament-based parent selection
       - RouletteSelection.select: Fitness-proportionate selection

    5. HIERARCHICAL GP (if max_depth > 0)
       - BooleanGP._create_child_populations: Initialize child populations
       - BooleanGP._generate_child_feedback: Aggregate child scores
       - PopulationGenerator.generate: Create initial population

    The decorator wraps each function to measure:
    - Number of calls
    - Total elapsed time (including child calls)
    - Own time (excluding child calls)
    """
    decorator = get_timed_decorator("GPTimer")

    # Rule evaluation - these are the hot paths
    Literal.evaluate = decorator(Literal.evaluate)
    Rule.flatten = decorator(Rule.flatten)
    Rule.__len__ = decorator(Rule.__len__)
    And.evaluate = decorator(And.evaluate)
    Or.evaluate = decorator(Or.evaluate)

    import hgp_lib

    hgp_lib.rules.utils.select_crossover_point = decorator(hgp_lib.rules.utils.select_crossover_point)
    hgp_lib.crossover.crossover_executor.select_crossover_point = decorator(hgp_lib.crossover.crossover_executor.select_crossover_point)
    hgp_lib.rules.utils.replace_with_rule = decorator(hgp_lib.rules.utils.replace_with_rule)
    hgp_lib.rules.utils.deep_swap = decorator(hgp_lib.rules.utils.deep_swap)
    hgp_lib.crossover.crossover_executor.deep_swap = decorator(hgp_lib.crossover.crossover_executor.deep_swap)

    np.random.randint = decorator(np.random.randint)
    random.choice = decorator(random.choice)
    random.random = decorator(random.random)

    AddLiteral.apply = decorator(AddLiteral.apply)
    PromoteLiteral.apply = decorator(PromoteLiteral.apply)
    ReplaceLiteral.apply = decorator(ReplaceLiteral.apply)
    NegateMutation.apply = decorator(NegateMutation.apply)
    DeleteMutation.apply = decorator(DeleteMutation.apply)
    ReplaceOperator.apply = decorator(ReplaceOperator.apply)
    RemoveIntermediateOperator.apply = decorator(RemoveIntermediateOperator.apply)

    # Scoring function
    global f1_score
    f1_score = decorator(f1_score)

    # GP algorithm core
    BooleanGP.step = decorator(BooleanGP.step)
    BooleanGP._new_generation = decorator(BooleanGP._new_generation)
    BooleanGP._evaluate_population = decorator(BooleanGP._evaluate_population)
    BooleanGP._update_best = decorator(BooleanGP._update_best)
    BooleanGP.validate_population = decorator(BooleanGP.validate_population)

    # Hierarchical GP operations
    BooleanGP._create_child_populations = decorator(BooleanGP._create_child_populations)
    BooleanGP._generate_child_feedback = decorator(BooleanGP._generate_child_feedback)

    # Genetic operators
    CrossoverExecutor.apply = decorator(CrossoverExecutor.apply)
    MutationExecutor.apply = decorator(MutationExecutor.apply)
    MutationExecutor._mutate = decorator(MutationExecutor._mutate)

    # Selection strategies
    TournamentSelection.select = decorator(TournamentSelection.select)

    # Population generation
    PopulationGenerator.generate = decorator(PopulationGenerator.generate)


def print_timing_results(measurements: Dict, args: argparse.Namespace) -> None:
    """
    Print formatted timing results sorted by own time.

    For each profiled function, shows:
    - Number of calls
    - Own time (excluding child calls) - most useful for identifying bottlenecks
    - Elapsed time (including child calls)
    - Per-call averages

    Args:
        measurements: Dictionary from setup_timing()
        args: Command line arguments for context
    """
    print("\n" + "=" * 80)
    print("TIMING RESULTS")
    print("=" * 80)
    print(
        f"Configuration: max_depth={args.max_depth}, epochs={args.num_epochs}, "
        f"pop_size={args.population_size}"
    )
    if args.max_depth > 0:
        print(
            f"               child_pops={args.num_child_populations}, "
            f"feature_frac={args.feature_fraction}"
        )
    print("-" * 80)

    # Sort by own time (descending) to show bottlenecks first
    sorted_measurements = sorted(
        measurements.items(),
        key=lambda x: x[1][2],  # Sort by own_time
        reverse=True,
    )

    # Calculate total time for percentage
    total_own_time = sum(v[2] for v in measurements.values()) / 1e9

    table = PrettyTable(
        [
            "Function",
            "Calls",
            "Own(s)",
            "Elapsed(s)",
            "Per-call(s)",
            "Per-call(elapsed_s)",
        ]
    )

    for key, (counts, elapsed, own_time) in sorted_measurements:
        own_time_s = round(own_time / 1e9, 4)
        elapsed_s = round(elapsed / 1e9, 4)
        per_call_s = round(own_time / counts / 1e9, 4)
        per_call_elapsed_s = round(elapsed / counts / 1e9, 4)
        print(
            f"Function {key} was called {counts} time(s) and took {own_time_s}s/{elapsed_s}s "
            f"({per_call_s}s/{per_call_elapsed_s}s per call)"
        )
        table.add_row(
            [key, counts, own_time_s, elapsed_s, per_call_s, per_call_elapsed_s]
        )

    print("-" * 80)
    print(f"TOTAL: {total_own_time:.4f}s")
    print()
    print(table)
    print()


# ==============================================================================
# RULE VALIDITY CHECK
# ==============================================================================


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
# MAIN TRAINING FUNCTION
# ==============================================================================


def main(args: argparse.Namespace) -> None:
    """
    Main training and profiling function.

    Workflow:
    1. Set up timing infrastructure
    2. Apply decorators to functions we want to profile
    3. Load and preprocess data
    4. Configure GP algorithm with specified parameters
    5. Run training via GPTrainer
    6. Evaluate on test set
    7. Print timing results
    """
    # Initialize profiling
    measurements = setup_timing()
    apply_timing_decorators()

    # Load data
    (
        train_data,
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels,
        feature_names,
    ) = preprocess_paysim_data(args.data_path)

    # Create validity checker
    is_valid = create_validity_checker(args.max_rule_size)

    # Configure GP algorithm
    # Note: Most parameters use sensible defaults from BooleanGPConfig
    print("\n" + "=" * 60)
    print("GP CONFIGURATION")
    print("=" * 60)

    population_generator = PopulationGenerator(
        strategies=[RandomStrategy(num_literals=train_data.shape[1])],
        population_size=args.population_size,
    )

    gp_config = BooleanGPConfig(
        train_data=train_data,
        train_labels=train_labels,
        score_fn=f1_score,
        check_valid=is_valid,
        optimize_scorer=args.optimize_scorer,
        regeneration=args.regeneration,
        regeneration_patience=args.regeneration_patience,
        population_generator=population_generator,
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
        val_data=val_data,
        val_labels=val_labels,
        val_every=args.val_every,
        progress_bar=not args.no_progress,
    )

    print(f"\nTraining epochs: {args.num_epochs}")
    print(f"Validation every: {args.val_every} epochs")

    # Initialize and run trainer
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    trainer = GPTrainer(trainer_config)
    result = trainer.fit()

    # Print training summary
    print("\n" + "-" * 60)
    print("Training Summary:")
    print(f"  Best train score: {result.best_score:.4f}")
    if result.val_history:
        val_scores = result.val_history.best_scores()
        print(f"  Best val score: {max(val_scores):.4f}")
        print(f"  Final val score: {val_scores[-1]:.4f}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)

    test_metrics = trainer.score(test_data, test_labels, score_fn=f1_score)
    print(f"Test F1 score: {test_metrics.best:.4f}")
    print("\nBest rule:")
    print(f"  {test_metrics.best_rule}")
    print("\nReadable form:")
    print(f"  {test_metrics.best_rule.to_str(feature_names)}")

    # Print profiling results
    print_timing_results(measurements, args)

    print("=" * 60)
    print("Profiling completed!")
    print("=" * 60)


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Arguments are organized into groups:
    - Data: Input file path
    - Training: Epochs, validation frequency
    - GP Algorithm: Population size, rule constraints
    - Hierarchical GP: Depth, child populations, feature sampling
    - Optimization: Scorer optimization, regeneration
    - Output: Progress bar control
    """
    parser = argparse.ArgumentParser(
        description="Profile Boolean GP training on PaySim data",
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

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of training epochs",
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

    # Output arguments
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar (cleaner output for logging)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
