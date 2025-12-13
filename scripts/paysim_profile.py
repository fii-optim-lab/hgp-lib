"""
PaySim Fraud Detection Training Script

This script trains a Boolean GP model on PaySim transaction data to detect fraud.
This script does not work with the original dataset file and needs preprocessing before.
TODO: We should add docs example with how to run on paysim
"""

import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from timed_decorator.builder import create_timed_decorator, get_timed_decorator

from hgp_lib.algorithms import BooleanGP
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.mutations import (
    MutationExecutor,
    create_standard_literal_mutations,
    create_standard_operator_mutations,
)
from hgp_lib.populations import PopulationGenerator, RandomStrategy
from hgp_lib.preprocessing import StandardBinarizer
from hgp_lib.rules import Rule, Literal, And, Or
from functools import partial


def fast_f1_score(y_pred, y_true, sample_weight=None):
    if sample_weight is None:
        y_pred_sum = y_pred.sum()
        y_true_sum = y_true.sum()
    else:
        y_pred_sum = np.dot(y_pred, sample_weight)
        y_true_sum = np.dot(y_true, sample_weight)

    if y_pred_sum == 0 or y_true_sum == 0:
        if y_pred_sum == 0 and y_true_sum == 0:
            return 1.0
        return 0.0

    if sample_weight is None:
        return 2 * (y_pred & y_true).sum() / (y_pred_sum + y_true_sum)
    return 2 * np.dot((y_pred & y_true), sample_weight) / (y_pred_sum + y_true_sum)


def preprocess_paysim_data(hdf_path: str):
    print(f"Loading data from {hdf_path}...")

    feature_columns = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    df: pd.DataFrame = pd.read_hdf(hdf_path)
    if "isFraud" in df.columns:
        target_column = "isFraud"
    elif "target" in df.columns:
        target_column = "target"
    else:
        raise RuntimeError(df.columns)

    data = df[feature_columns].copy()
    labels = df[target_column].values.copy()

    del df

    print(f"Loaded {len(data)} samples with {len(feature_columns)} features")
    print(f"Fraud rate: {labels.mean():.4f} ({labels.sum()} fraud cases)")

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

    print(f"Train: {len(train_data)} samples")
    print(f"Val: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples")

    print("\nBinarizing data...")
    binarizer = StandardBinarizer(num_bins=5)
    train_data_bin = binarizer.fit_transform(train_data, train_labels)
    val_data_bin = binarizer.transform(val_data)
    test_data_bin = binarizer.transform(test_data)

    print(f"Binarized features: {train_data_bin.shape[1]} (from {train_data.shape[1]})")

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
    )


def setup_timing():
    measurements = {}
    create_timed_decorator(
        "GPTimer",
        nested=True,
        collect_gc=False,
        disable_gc=True,
        stdout=False,
        out=measurements,
    )
    return measurements


def apply_timing_decorators(gp_algo: BooleanGP, score_fn):
    decorator = get_timed_decorator("GPTimer")

    Literal.evaluate = decorator(Literal.evaluate)
    And.evaluate = decorator(And.evaluate)
    Or.evaluate = decorator(Or.evaluate)

    score_fn = decorator(score_fn)

    gp_algo.step = decorator(gp_algo.step)
    gp_algo._new_generation = decorator(gp_algo._new_generation)
    gp_algo._evaluate_population = decorator(gp_algo._evaluate_population)
    gp_algo._update_best = decorator(gp_algo._update_best)
    gp_algo.validate_population = decorator(gp_algo.validate_population)

    gp_algo.crossover_executor.apply = decorator(gp_algo.crossover_executor.apply)
    gp_algo.mutation_executor.apply = decorator(gp_algo.mutation_executor.apply)
    gp_algo.selection.select = decorator(gp_algo.selection.select)

    return score_fn


def print_timing_results(measurements: dict):
    print("\n" + "=" * 60)
    print("Timing Results:")
    print("=" * 60)
    sorted_measurements = sorted(measurements.items(), key=lambda x: x[1][2])

    for key, (counts, elapsed, own_time) in sorted_measurements:
        own_time /= 1e9
        elapsed /= 1e9
        print(
            f"Function {key} was called {counts} time(s) and took {own_time:.4f}s/{elapsed:.4f}s "
            f"({own_time / counts:.4f}/{elapsed / counts:.4f} per call)"
        )

    print()


def remove_duplicates(data, labels):
    Xy = np.hstack((data, labels[:, None]))
    Xy_unique, sample_weight = np.unique(Xy, axis=0, return_counts=True)
    return Xy_unique[:, :-1], Xy_unique[:, -1], sample_weight


def main():
    measurements = setup_timing()

    hdf_path = "data/PaySim.hdf"
    num_epochs = 5000
    population_size = 100
    max_rule_size = 50
    mutation_p = 0.05
    crossover_p = 0.7

    (
        train_data_bin,
        train_labels,
        val_data_bin,
        val_labels,
        test_data_bin,
        test_labels,
    ) = preprocess_paysim_data(hdf_path)
    train_data_bin, train_labels, sample_weight = remove_duplicates(
        train_data_bin, train_labels
    )
    val_data_bin, val_labels, sample_weight_val = remove_duplicates(
        train_data_bin, train_labels
    )

    train_score_fn = partial(fast_f1_score, sample_weight=sample_weight)
    val_score_fn = partial(fast_f1_score, sample_weight=sample_weight_val)
    test_score_fn = fast_f1_score

    def is_rule_valid(rule: Rule) -> bool:
        if len(rule) > max_rule_size:
            return False
        return True

    num_features = train_data_bin.shape[1]
    literal_mutations = create_standard_literal_mutations(num_features)
    operator_mutations = create_standard_operator_mutations(num_features)

    random_strategy = RandomStrategy(num_literals=num_features)
    population_generator = PopulationGenerator(
        strategies=[random_strategy],
        population_size=population_size,
    )

    mutation_executor = MutationExecutor(
        literal_mutations=literal_mutations,
        operator_mutations=operator_mutations,
        mutation_p=mutation_p,
        check_valid=is_rule_valid,
        num_tries=5,
    )

    crossover_executor = CrossoverExecutor(
        crossover_p=crossover_p,
        check_valid=is_rule_valid,
        num_tries=2,
    )

    print("\nInitializing Boolean GP...")
    gp_algo = BooleanGP(
        score_fn=train_score_fn,
        population_generator=population_generator,
        mutation_executor=mutation_executor,
        crossover_executor=crossover_executor,
        regeneration=True,
        regeneration_patience=200,
    )

    train_score_fn = apply_timing_decorators(gp_algo, train_score_fn)
    gp_algo.train_score_fn = train_score_fn

    print(f"\nStarting training for {num_epochs} epochs...")

    val_best = 0
    val_avg = 0
    with tqdm(range(num_epochs), desc="Training") as tbar:
        for epoch in tbar:
            train_metrics = gp_algo.step(train_data_bin, train_labels)

            if (epoch + 1) % 10 == 0:
                val_metrics = gp_algo.validate_population(
                    val_data_bin, val_labels, val_score_fn
                )
                val_best = val_metrics["best"]
                val_avg = val_metrics["population_scores"].mean()
            tbar.set_postfix(
                {
                    "current_best": f"{train_metrics['current_best']:.4f}",
                    "train_best": f"{train_metrics['best']:.4f}",
                    "real_best": f"{train_metrics['real_best']:.4f}",
                    "val_best": f"{val_best:.4f}",
                    "val_avg": f"{val_avg:.4f}",
                }
            )

    print("\n" + "-" * 60)
    print("Final evaluation on test set...")
    test_metrics = gp_algo.validate_population(
        test_data_bin, test_labels, test_score_fn, all_time_best=True
    )
    print(f"Test best: {test_metrics['best']:.4f}")
    print(f"Test population average: {test_metrics['population_scores'].mean():.4f}")

    print_timing_results(measurements)
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
