"""
This script profiles the training of a Boolean GP model.


==============================================================================
USAGE EXAMPLES
==============================================================================

Basic profiling (standard GP):
    python scripts/profile_trainer.py --data_path data/PaySim.hdf --num_epochs 500

Profile hierarchical GP with depth 1:
    python scripts/profile_trainer.py --data_path data/PaySim.hdf --max_depth 1 --num_epochs 500

Compare depths with same epoch budget:
    python scripts/profile_trainer.py --data_path data/PaySim.hdf --max_depth 0 --num_epochs 1000
    python scripts/profile_trainer.py --data_path data/PaySim.hdf --max_depth 1 --num_epochs 1000
    python scripts/profile_trainer.py --data_path data/PaySim.hdf --max_depth 2 --num_epochs 1000

Adjust population and feature sampling:
    python scripts/profile_trainer.py --data_path data/PaySim.hdf --max_depth 1 --num_child_populations 5 --feature_fraction 0.5
"""

from prettytable import PrettyTable
import argparse
import gc
from typing import Dict, Tuple

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
    PopulationGeneratorFactory,
)
from hgp_lib.preprocessing import StandardBinarizer, load_data
from hgp_lib.rules import Rule
from hgp_lib.selections import TournamentSelection
from hgp_lib.trainers import GPTrainer
from hgp_lib.utils import ComplexityCheck
from hgp_lib.utils.metrics import fast_f1_score


def preprocess_data(data_path: str, num_bins: int = 5) -> Tuple:
    """
    Load and preprocess data for training.

    This function:
    1. Loads the HDF file containing transaction data
    2. Splits into train/val/test sets
    3. Binarizes features using StandardBinarizer

    The binarization is crucial for Boolean GP - it converts continuous features
    into boolean features that can be used as literals in the evolved rules.

    Args:
        data_path: Path to the preprocessed PaySim HDF file
        num_bins: Number of bins for binarization

    Returns:
        Tuple of (train_data, train_labels, val_data, val_labels,
                  test_data, test_labels, feature_names)
    """
    print(f"Loading data from {data_path}...")

    data, labels = load_data(data_path)

    print(f"Loaded {len(data)} samples with {len(data.columns)} features")
    print(f"Positive rate: {labels.mean():.4f} ({labels.sum()} positive cases)")

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

    print(f"Train: {len(train_data)} samples ({train_labels.sum()} positive)")
    print(f"Val:   {len(val_data)} samples ({val_labels.sum()} positive)")
    print(f"Test:  {len(test_data)} samples ({test_labels.sum()} positive)")

    # Binarize features - converts continuous to boolean
    # Each feature becomes num_bins boolean features
    print(f"\nBinarizing features (num_bins={num_bins})...")
    binarizer = StandardBinarizer(num_bins=num_bins)
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


def setup_timing() -> Dict:
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
    decorator = get_timed_decorator("GPTimer")

    # Rule evaluation - these are the hot paths
    # Literal.evaluate = decorator(Literal.evaluate)
    Rule.flatten = decorator(Rule.flatten)
    # Rule.copy = decorator(Rule.copy)  # Overhead!
    # Rule.__len__ = decorator(Rule.__len__)  # Overhead!
    # And.evaluate = decorator(And.evaluate)
    # Or.evaluate = decorator(Or.evaluate)

    import hgp_lib

    hgp_lib.rules.utils.select_crossover_point = decorator(
        hgp_lib.rules.utils.select_crossover_point
    )
    hgp_lib.crossover.crossover_executor.select_crossover_point = decorator(
        hgp_lib.crossover.crossover_executor.select_crossover_point
    )
    hgp_lib.rules.utils.replace_with_rule = decorator(
        hgp_lib.rules.utils.replace_with_rule
    )
    hgp_lib.rules.utils.deep_swap = decorator(hgp_lib.rules.utils.deep_swap)
    hgp_lib.crossover.crossover_executor.deep_swap = decorator(
        hgp_lib.crossover.crossover_executor.deep_swap
    )

    # np.random.randint = decorator(np.random.randint)
    # random.choice = decorator(random.choice)
    # random.random = decorator(random.random)

    AddLiteral.apply = decorator(AddLiteral.apply)
    PromoteLiteral.apply = decorator(PromoteLiteral.apply)
    ReplaceLiteral.apply = decorator(ReplaceLiteral.apply)
    NegateMutation.apply = decorator(NegateMutation.apply)
    DeleteMutation.apply = decorator(DeleteMutation.apply)
    ReplaceOperator.apply = decorator(ReplaceOperator.apply)
    RemoveIntermediateOperator.apply = decorator(RemoveIntermediateOperator.apply)

    # GP algorithm core
    BooleanGP.step = decorator(BooleanGP.step)
    BooleanGP._new_generation = decorator(BooleanGP._new_generation)
    BooleanGP.evaluate_population = decorator(BooleanGP.evaluate_population)
    BooleanGP._update_best = decorator(BooleanGP._update_best)

    # Hierarchical GP operations
    BooleanGP._create_child_populations = decorator(BooleanGP._create_child_populations)
    BooleanGP._generate_child_feedback = decorator(BooleanGP._generate_child_feedback)

    # Genetic operators
    CrossoverExecutor.apply = decorator(CrossoverExecutor.apply)
    CrossoverExecutor.crossover = decorator(CrossoverExecutor.crossover)
    MutationExecutor.apply = decorator(MutationExecutor.apply)
    MutationExecutor._mutate = decorator(MutationExecutor._mutate)
    FeatureSamplingStrategy.sample = decorator(FeatureSamplingStrategy.sample)

    # Selection strategies
    TournamentSelection.select = decorator(TournamentSelection.select)

    # Population generation
    PopulationGenerator.generate = decorator(PopulationGenerator.generate)


def print_timing_results(measurements: Dict, args: argparse.Namespace) -> None:
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


def main(args: argparse.Namespace) -> None:
    measurements = setup_timing()
    apply_timing_decorators()

    (
        train_data,
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels,
        feature_names,
    ) = preprocess_data(args.data_path, num_bins=args.num_bins)

    is_valid = ComplexityCheck(args.max_rule_size)
    gp_config = BooleanGPConfig(
        train_data=train_data,
        train_labels=train_labels,
        score_fn=fast_f1_score,
        check_valid=is_valid,
        optimize_scorer=args.optimize_scorer,
        regeneration=args.regeneration,
        regeneration_patience=args.regeneration_patience,
        population_factory=PopulationGeneratorFactory(
            population_size=args.population_size
        ),
    )

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

    trainer = GPTrainer(trainer_config)
    result = trainer.fit()

    print("\n" + "-" * 60)
    print("Training Summary:")
    best_train = max(gen.best_train_score for gen in result.generations)
    print(f"  Best train score: {best_train:.4f}")
    if result.best_val_score is not None:
        print(f"  Best val score: {result.best_val_score:.4f}")

    test_score = trainer.gp_algo.evaluate_best(
        test_data, test_labels, score_fn=fast_f1_score
    )
    print(f"Test F1 score: {test_score:.4f}")
    print("\nBest rule:")
    print(f"  {result.global_best_rule}")
    print("\nReadable form:")
    print(f"  {result.global_best_rule.to_str(feature_names)}")

    print_timing_results(measurements, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Boolean GP training on PaySim data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data_path",
        type=str,
        required=True,
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
        "--no_optimize_scorer",
        action="store_false",
        dest="optimize_scorer",
        help="Disable scorer optimization",
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
