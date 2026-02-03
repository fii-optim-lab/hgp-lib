"""
PaySim Fraud Detection Benchmarking Script

Demonstrates how to benchmark Boolean GP on PaySim transaction data using
multi-run evaluation with k-fold cross-validation. Uses default hyperparameters
with scorer optimization enabled.

Usage:
    python scripts/paysim_benchmark.py

Requires preprocessed PaySim data in HDF format at data/PaySim.hdf
"""

import gc

import numpy as np
import pandas as pd

from hgp_lib import BenchmarkerConfig
from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.preprocessing import StandardBinarizer


def f1_score(y_pred, y_true, sample_weight=None):
    """F1 score supporting sample_weight for scorer optimization."""
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


def load_and_binarize_paysim(hdf_path: str):
    """Load PaySim data and binarize features."""
    print(f"Loading data from {hdf_path}...")
    df: pd.DataFrame = pd.read_hdf(hdf_path)

    # Detect target column
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

    # Binarize features
    print("\nBinarizing features...")
    binarizer = StandardBinarizer(num_bins=5)
    data_bin = binarizer.fit_transform(data, labels)
    feature_names = {i: col for i, col in enumerate(data_bin.columns)}

    print(f"Binarized: {data_bin.shape[1]} features (from {data.shape[1]})")

    data_bin = data_bin.values
    del data
    gc.collect()

    return data_bin, labels, feature_names


def main():
    hdf_path = "data/PaySim.hdf"
    num_epochs = 1000

    data, labels, feature_names = load_and_binarize_paysim(hdf_path)

    # Configure benchmarker with defaults
    # - 30 runs with different random seeds
    # - 5-fold CV per run
    # - 20% held out for test
    # - Scorer optimization enabled (deduplicates data, uses sample weights)
    config = BenchmarkerConfig(
        data=data,
        labels=labels,
        score_fn=f1_score,
        num_epochs=num_epochs,
        # All other parameters use defaults:
        # num_runs=30, n_folds=5, test_size=0.2, optimize_scorer=True
    )

    print("\nBenchmark configuration:")
    print(f"  Epochs per fold: {config.num_epochs}")
    print(f"  Runs: {config.num_runs}")
    print(f"  Folds per run: {config.n_folds}")
    print(f"  Test size: {config.test_size}")
    print(f"  Scorer optimization: {config.optimize_scorer}")

    benchmarker = GPBenchmarker(config)

    print("\nRunning benchmark...")
    result = benchmarker.fit()

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(
        f"Test F1 Score:  {result.mean_test_score:.4f} +/- {result.std_test_score:.4f}"
    )
    print(
        f"Val F1 Score:   {result.mean_best_val_score:.4f} +/- {result.std_best_val_score:.4f}"
    )

    print("\nPer-run test scores:")
    for i, score in enumerate(result.all_test_scores):
        print(f"  Run {i:2d}: {score:.4f}")

    # Show best rule from run with highest test score
    best_run_idx = np.argmax(result.all_test_scores)
    best_rule = result.all_best_rules[best_run_idx]
    print(f"\nBest rule (from run {best_run_idx}):")
    print(f"  {best_rule}")
    print("\nReadable form:")
    print(f"  {best_rule.to_str(feature_names)}")

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
