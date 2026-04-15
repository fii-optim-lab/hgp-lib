import argparse
import random
import subprocess
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from time import sleep

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score

from hgp_lib.preprocessing import load_data
from preprocess.pmlb_preprocess import save_pmlb_data
from tqdm.contrib.concurrent import process_map


def get_binary_classification_datasets():
    from pmlb.dataset_lists import df_summary

    return df_summary[df_summary["n_classes"] == 2.0]["dataset"].tolist()


def get_commands_for_datasets(dataset_names, n_runs, n_folds, data_dir):
    commands = []
    for dataset_name in dataset_names:
        cmd = (
            f"{sys.executable} "
            "scripts/optuna_hypertuning.py "
            f"--data-path {data_dir}/{dataset_name}.hdf "
            "--n-trials 25 "
            "--max-n-trials 25 "
            f"--n-runs {n_runs} "
            f"--n-folds {n_folds} "
            f"--study-name pmlb_{dataset_name} "
            "--artifact-dir ./artifacts"
        )
        commands.append(cmd)
    return commands


def subprocess_runner(command: str):
    sleep(random.randint(1, 5))
    subprocess.run(command.split(" "), check=True)


def run_dt_benchmark(
    dataset_name: str, n_runs: int = 30, n_folds: int = 5, data_dir: str = "data"
) -> dict:
    data_path = f"{data_dir}/{dataset_name}.hdf"
    if not os.path.isfile(data_path):
        try:
            save_pmlb_data(dataset_name, data_dir)
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            return None

    try:
        data, labels = load_data(data_path)
    except Exception as e:
        print(f"Failed to read {dataset_name}: {e}")
        return None

    test_scores = []

    for run_id in range(n_runs):
        seed = 42 + run_id
        np.random.seed(seed)
        random.seed(seed)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, stratify=labels, random_state=seed
        )

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        best_val_score = -float("inf")
        best_model = None

        # K-fold CV
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train[val_idx]

            dt = DecisionTreeClassifier(random_state=seed)
            dt.fit(X_fold_train, y_fold_train)

            preds = dt.predict(X_fold_val)
            val_score = f1_score(y_fold_val, preds)

            if val_score > best_val_score:
                best_val_score = val_score
                best_model = dt

        # Evaluate best model on test set
        test_preds = best_model.predict(X_test)
        test_score = f1_score(y_test, test_preds)
        test_scores.append(test_score)

    return {
        "dataset": dataset_name,
        "mean_test_score": np.mean(test_scores),
        "std_test_score": np.std(test_scores),
    }


def _run_dt_wrapper(args_tuple):
    dataset_name, n_runs, n_folds, data_dir = args_tuple
    return run_dt_benchmark(
        dataset_name, n_runs=n_runs, n_folds=n_folds, data_dir=data_dir
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_jobs", type=int, default=4)
    parser.add_argument(
        "--model",
        type=str,
        choices=["gp", "dt"],
        default="gp",
        help="Model to run: 'gp' (default) or 'dt' (Decision Tree).",
    )
    parser.add_argument(
        "--n-runs", type=int, default=5, help="Number of runs for DT benchmark."
    )
    parser.add_argument(
        "--n-folds", type=int, default=3, help="Number of folds for DT benchmark."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory for storing datasets."
    )
    args = parser.parse_args()

    dataset_names = get_binary_classification_datasets()

    if args.model == "gp":
        commands = get_commands_for_datasets(
            dataset_names, args.n_runs, args.n_folds, args.data_dir
        )
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            executor.map(subprocess_runner, commands)
    elif args.model == "dt":
        os.makedirs(args.data_dir, exist_ok=True)

        args_list = [
            (name, args.n_runs, args.n_folds, args.data_dir) for name in dataset_names
        ]

        results = process_map(
            _run_dt_wrapper,
            args_list,
            max_workers=args.n_jobs,
            desc="Running DT benchmarks",
            chunksize=1,
        )

        results = [res for res in results if res is not None]

        df_results = pd.DataFrame(results)
        df_results.to_csv("pmlb_dt.csv", index=False)
        print("Saved results to pmlb_dt.csv")


if __name__ == "__main__":
    main()
