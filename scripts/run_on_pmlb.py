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


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.compose import make_column_selector

from hgp_lib.preprocessing import StandardBinarizer


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


def run_sklearn_benchmark(
    dataset_name: str,
    model_type: str,
    n_runs: int = 30,
    n_folds: int = 5,
    data_dir: str = "data",
    max_leaf_nodes: int = None,
    binarizer_type: str = "standard",
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
        best_discretizer = None

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train[val_idx]

            discretizer = None
            if model_type == "boolxai":
                if binarizer_type == "standard":
                    discretizer = StandardBinarizer(num_bins=5)
                    X_fold_train = discretizer.fit_transform(
                        X_fold_train, y_fold_train
                    ).to_numpy(dtype=bool)
                    X_fold_val = discretizer.transform(X_fold_val).to_numpy(dtype=bool)
                elif binarizer_type == "sklearn":
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        discretizer = make_column_transformer(
                            (
                                KBinsDiscretizer(
                                    n_bins=5, encode="onehot-dense", strategy="quantile"
                                ),
                                make_column_selector(dtype_include=np.number),
                            ),
                            (
                                OneHotEncoder(
                                    sparse_output=False, handle_unknown="ignore"
                                ),
                                make_column_selector(
                                    dtype_include=[object, "category"]
                                ),
                            ),
                            remainder="passthrough",
                        )
                        X_fold_train = discretizer.fit_transform(X_fold_train)
                        X_fold_val = discretizer.transform(X_fold_val)
                else:
                    raise ValueError(f"Unknown binarizer type: {binarizer_type}")

            if model_type == "dt":
                model = DecisionTreeClassifier(
                    random_state=seed, max_leaf_nodes=max_leaf_nodes
                )
            elif model_type == "boolxai":
                from boolxai.classifiers import RuleClassifier

                model = RuleClassifier(random_state=seed, num_jobs=1)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.fit(X_fold_train, y_fold_train)

            preds = model.predict(X_fold_val)
            val_score = f1_score(y_fold_val, preds)

            if val_score > best_val_score:
                best_val_score = val_score
                best_model = model
                best_discretizer = discretizer

        # Evaluate best model on test set
        if model_type == "boolxai" and best_discretizer is not None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if binarizer_type == "standard":
                    X_test_transformed = best_discretizer.transform(X_test).to_numpy(
                        dtype=bool
                    )
                else:
                    X_test_transformed = best_discretizer.transform(X_test)
        else:
            X_test_transformed = X_test

        test_preds = best_model.predict(X_test_transformed)
        test_score = f1_score(y_test, test_preds)
        test_scores.append(test_score)

    return {
        "dataset": dataset_name,
        "mean_test_score": np.mean(test_scores),
        "std_test_score": np.std(test_scores),
    }


def _run_sklearn_wrapper(args_tuple):
    (
        dataset_name,
        model_type,
        n_runs,
        n_folds,
        data_dir,
        max_leaf_nodes,
        binarizer_type,
    ) = args_tuple
    return run_sklearn_benchmark(
        dataset_name,
        model_type=model_type,
        n_runs=n_runs,
        n_folds=n_folds,
        data_dir=data_dir,
        max_leaf_nodes=max_leaf_nodes,
        binarizer_type=binarizer_type,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_jobs", type=int, default=4)
    parser.add_argument(
        "--model",
        type=str,
        choices=["gp", "dt", "boolxai"],
        default="gp",
        help="Model to run: 'gp' (default), 'dt' (Decision Tree), or 'boolxai'.",
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
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=None,
        help="Maximum number of leaf nodes for Decision Tree.",
    )
    parser.add_argument(
        "--binarizer",
        type=str,
        choices=["standard", "sklearn"],
        default="standard",
        help="Binarizer type to use for boolxai.",
    )
    args = parser.parse_args()

    dataset_names = get_binary_classification_datasets()

    if args.model == "gp":
        commands = get_commands_for_datasets(
            dataset_names, args.n_runs, args.n_folds, args.data_dir
        )
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            executor.map(subprocess_runner, commands)
    elif args.model in ["dt", "boolxai"]:
        os.makedirs(args.data_dir, exist_ok=True)

        args_list = [
            (
                name,
                args.model,
                args.n_runs,
                args.n_folds,
                args.data_dir,
                args.max_leaf_nodes,
                args.binarizer,
            )
            for name in dataset_names
        ]

        results = process_map(
            _run_sklearn_wrapper,
            args_list,
            max_workers=args.n_jobs,
            desc=f"Running {args.model.upper()} benchmarks",
            chunksize=1,
        )

        results = [res for res in results if res is not None]

        df_results = pd.DataFrame(results)

        if args.model == "dt" and args.max_leaf_nodes is not None:
            csv_filename = f"pmlb_dt_{args.max_leaf_nodes}.csv"
        elif args.model == "boolxai":
            csv_filename = f"pmlb_boolxai_{args.binarizer}.csv"
        else:
            csv_filename = f"pmlb_{args.model}.csv"

        df_results.to_csv(csv_filename, index=False)
        print(f"Saved results to {csv_filename}")


if __name__ == "__main__":
    main()
