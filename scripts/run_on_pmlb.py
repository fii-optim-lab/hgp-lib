import argparse
import os
import random
import subprocess
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from time import sleep

import numpy as np
import pandas as pd
from preprocess.pmlb_preprocess import save_pmlb_data
from rule_conversion import boolxai_to_rule, dt_to_rule, ripper_to_rule, rule_stats
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from tqdm.contrib.concurrent import process_map

from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.populations import FeatureSamplingStrategy
from hgp_lib.preprocessing import StandardBinarizer
from hgp_lib.utils.metrics import fast_f1_score

# Run i of every method uses seed BASE_SEED + i for its stratified train/test split and
# k-fold split, so all methods see the same splits. 0 is BenchmarkerConfig's default.
BASE_SEED = 0
TEST_SIZE = 0.2


def get_binary_classification_datasets():
    from pmlb.dataset_lists import df_summary

    return df_summary[df_summary["n_classes"] == 2.0]["dataset"].tolist()


def get_commands_for_datasets(dataset_names, n_runs, n_folds, data_dir):
    commands = []
    for dataset_name in dataset_names:
        cmd = (
            f"{sys.executable} "
            "scripts/optuna_hypertuning.py "
            f"--data_path {data_dir}/{dataset_name}.hdf "
            "--n_trials 50 "
            "--max_n_trials 100 "
            f"--n_runs {n_runs} "
            f"--n_folds {n_folds} "
            f"--study_name pmlb_{dataset_name} "
            "--artifact_dir ./artifacts"
        )
        commands.append(cmd)
    return commands


def subprocess_runner(command: str):
    sleep(random.randint(1, 5))
    subprocess.run(
        command.split(" "), check=True, env={**os.environ, "PYTHONOPTIMIZE": "2"}
    )


def load_dataset(dataset_name: str, data_dir: str):
    """
    Load a PMLB dataset as ``(features, labels)``, downloading it if needed.

    ``hgp_lib.preprocessing.load_data`` casts the target to ``bool``, which turns the
    ``{1, 2}`` targets of haberman and hepatitis into all-positive labels. Here the
    larger of the two class values is the positive class instead. This is identical to
    the bool cast for ``{0, 1}`` and ``{0, 2}`` targets.

    Returns:
        tuple[pd.DataFrame, np.ndarray] | None: The features and boolean labels, or
        ``None`` if the dataset could not be downloaded.
    """
    data_path = f"{data_dir}/{dataset_name}.hdf"
    if not os.path.isfile(data_path):
        try:
            save_pmlb_data(dataset_name, data_dir)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load {dataset_name}: {e}")
            return None

    data = pd.read_hdf(data_path)
    target = data.pop("target")
    classes = np.sort(target.unique())
    if len(classes) != 2:
        raise ValueError(
            f"{dataset_name}: expected 2 classes, found {classes.tolist()}"
        )
    labels = (target == classes[1]).to_numpy()
    return data, labels


def _summarize(test_scores, rule_sizes, n_literals) -> dict:
    return {
        "mean_test_score": np.mean(test_scores),
        "std_test_score": np.std(test_scores),
        "mean_rule_size": np.mean(rule_sizes),
        "std_rule_size": np.std(rule_sizes),
        "mean_n_literals": np.mean(n_literals),
        "std_n_literals": np.std(n_literals),
    }


def _make_sklearn_binarizer():
    return make_column_transformer(
        (
            KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile"),
            make_column_selector(dtype_include=np.number),
        ),
        (
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            make_column_selector(dtype_include=[object, "category"]),
        ),
        remainder="passthrough",
    )


def _binarize(binarizer, binarizer_type: str, model_type: str, X, y=None):
    """Fit and transform when ``y`` is given, otherwise only transform."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = binarizer.transform(X) if y is None else binarizer.fit_transform(X, y)
    # RIPPER keeps the DataFrame so its rules use the binarized feature names.
    if binarizer_type == "standard" and model_type == "boolxai":
        X = X.to_numpy(dtype=bool)
    return X


def run_sklearn_benchmark(
    data: pd.DataFrame,
    labels: np.ndarray,
    model_type: str,
    n_runs: int = 5,
    n_folds: int = 3,
    base_seed: int = BASE_SEED,
    max_leaf_nodes: int | None = None,
    binarizer_type: str = "standard",
) -> dict:
    """
    Benchmark DT, BoolXAI, or RIPPER with the protocol of ``GPBenchmarker``.

    Each run makes the same stratified train/test split and k-fold split as
    ``GPBenchmarker`` with the same seed, fits one model per fold, and scores the fold
    model with the best validation F1 on the test set. Rule size and literals are
    recorded for the model of every fold.
    """
    test_scores = []
    rule_sizes = []
    n_literals = []

    for run_id in range(n_runs):
        seed = base_seed + run_id
        np.random.seed(seed)
        random.seed(seed)

        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=TEST_SIZE, stratify=labels, random_state=seed
        )

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        best_val_score = -float("inf")
        best_model = None
        best_binarizer = None

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train[val_idx]

            binarizer = None
            if model_type in ("boolxai", "ripper") and binarizer_type != "none":
                if binarizer_type == "standard":
                    binarizer = StandardBinarizer(num_bins=5)
                    binarizer.progress_bar = False
                elif binarizer_type == "sklearn":
                    binarizer = _make_sklearn_binarizer()
                else:
                    raise ValueError(f"Unknown binarizer type: {binarizer_type}")
                X_fold_train = _binarize(
                    binarizer, binarizer_type, model_type, X_fold_train, y_fold_train
                )
                X_fold_val = _binarize(
                    binarizer, binarizer_type, model_type, X_fold_val
                )

            if model_type == "dt":
                model = DecisionTreeClassifier(
                    random_state=seed, max_leaf_nodes=max_leaf_nodes
                )
                model.fit(X_fold_train, y_fold_train)
                rule = dt_to_rule(model)
            elif model_type == "boolxai":
                from boolxai.classifiers import RuleClassifier

                model = RuleClassifier(random_state=seed, num_jobs=1)
                model.fit(X_fold_train, y_fold_train)
                rule = boolxai_to_rule(model.best_rule_)
            elif model_type == "ripper":
                import wittgenstein as lw

                model = lw.RIPPER(random_state=seed)
                # Labels are boolean; learn the rule set for the positive class.
                model.fit(X_fold_train, y_fold_train, pos_class=True)
                rule = ripper_to_rule(model)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            size, literals = rule_stats(rule)
            rule_sizes.append(size)
            n_literals.append(literals)

            preds = np.asarray(model.predict(X_fold_val), dtype=bool)
            val_score = f1_score(y_fold_val, preds)

            if val_score > best_val_score:
                best_val_score = val_score
                best_model = model
                best_binarizer = binarizer

        if best_binarizer is not None:
            X_test = _binarize(best_binarizer, binarizer_type, model_type, X_test)

        test_preds = np.asarray(best_model.predict(X_test), dtype=bool)
        test_scores.append(f1_score(y_test, test_preds))

    return _summarize(test_scores, rule_sizes, n_literals)


def run_gp_default_benchmark(
    data: pd.DataFrame,
    labels: np.ndarray,
    n_runs: int = 5,
    n_folds: int = 3,
    base_seed: int = BASE_SEED,
    num_epochs: int = 1000,
    max_depth: int = 0,
    num_child_populations: int = 3,
    feature_fraction: float = 0.33,
) -> dict:
    """
    Benchmark hgp-lib with default parameters, optionally with one or more levels of
    child populations. Rule size and literals are recorded for the best rule of every
    fold.
    """
    hierarchy = {}
    if max_depth > 0:
        hierarchy = {
            "max_depth": max_depth,
            "num_child_populations": num_child_populations,
            "sampling_strategy": FeatureSamplingStrategy(
                feature_fraction=feature_fraction
            ),
        }
    gp_config = BooleanGPConfig(score_fn=fast_f1_score, **hierarchy)
    trainer_config = TrainerConfig(
        gp_config=gp_config,
        num_epochs=num_epochs,
    )
    config = BenchmarkerConfig(
        data=data,
        labels=labels,
        trainer_config=trainer_config,
        num_runs=n_runs,
        n_folds=n_folds,
        test_size=TEST_SIZE,
        base_seed=base_seed,
        n_jobs=1,
        show_run_progress=False,
        show_fold_progress=False,
        show_epoch_progress=False,
    )

    result = GPBenchmarker(config).fit()

    stats = [
        rule_stats(fold.global_best_rule) for run in result.runs for fold in run.folds
    ]
    return _summarize(
        result.test_scores, [size for size, _ in stats], [lit for _, lit in stats]
    )


def run_dataset(task: dict) -> dict | None:
    """
    Benchmark one model on one dataset. Runs in a worker process.

    Returns:
        dict | None: The result row, with the runtime in seconds of the whole benchmark
        on this dataset (all runs and folds, including binarization). ``None`` if the
        dataset could not be downloaded. If the benchmark raises, the row holds the
        traceback under ``"error"``.
    """
    dataset_name = task["dataset"]
    try:
        loaded = load_dataset(dataset_name, task["data_dir"])
        if loaded is None:
            return None
        data, labels = loaded

        start = time.perf_counter()
        if task["model"] == "gp_default":
            result = run_gp_default_benchmark(data, labels, **task["kwargs"])
        else:
            result = run_sklearn_benchmark(
                data, labels, model_type=task["model"], **task["kwargs"]
            )
        runtime = time.perf_counter() - start
    except Exception:  # noqa: BLE001
        return {"dataset": dataset_name, "error": traceback.format_exc()}

    return {"dataset": dataset_name, **result, "runtime_s": runtime}


def default_csv_name(args) -> str:
    if args.model == "gp_default":
        name = f"pmlb_gp_default_epochs{args.num_epochs}"
        if args.max_depth > 0:
            name += f"_depth{args.max_depth}"
        return f"{name}.csv"
    if args.model == "dt" and args.max_leaf_nodes is not None:
        return f"pmlb_dt_{args.max_leaf_nodes}.csv"
    if args.model == "boolxai":
        return f"pmlb_boolxai_{args.binarizer}.csv"
    if args.model == "ripper":
        if args.binarizer == "none":
            return "pmlb_ripper.csv"
        return f"pmlb_ripper_{args.binarizer}.csv"
    return f"pmlb_{args.model}.csv"


def run_benchmarks(args, dataset_names: list[str]) -> None:
    """Benchmark one model on every dataset, one dataset per worker, and save a CSV."""
    os.makedirs(args.data_dir, exist_ok=True)

    common = {
        "n_runs": args.n_runs,
        "n_folds": args.n_folds,
        "base_seed": args.base_seed,
    }
    if args.model == "gp_default":
        kwargs = {
            **common,
            "num_epochs": args.num_epochs,
            "max_depth": args.max_depth,
            "num_child_populations": args.num_child_populations,
            "feature_fraction": args.feature_fraction,
        }
    else:
        kwargs = {
            **common,
            "max_leaf_nodes": args.max_leaf_nodes,
            "binarizer_type": args.binarizer,
        }

    def data_size(name):
        path = f"{args.data_dir}/{name}.hdf"
        return os.path.getsize(path) if os.path.isfile(path) else 0

    # Start the largest datasets first so they do not finish last on a single core.
    ordered = sorted(dataset_names, key=data_size, reverse=True)
    tasks = [
        {
            "dataset": name,
            "model": args.model,
            "data_dir": args.data_dir,
            "kwargs": kwargs,
        }
        for name in ordered
    ]

    results = process_map(
        run_dataset,
        tasks,
        max_workers=max(1, min(args.n_jobs, len(tasks))),
        desc=f"Running {args.model.upper()} benchmarks",
        chunksize=1,
    )

    rows = [res for res in results if res is not None and "error" not in res]
    errors = [res for res in results if res is not None and "error" in res]

    csv_filename = args.output or default_csv_name(args)
    out_dir = os.path.dirname(csv_filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_results = pd.DataFrame(rows)
    if not df_results.empty:
        df_results = df_results.sort_values("dataset").reset_index(drop=True)

    if errors:
        for res in errors:
            print(f"\n=== {res['dataset']} failed:\n{res['error']}", file=sys.stderr)
        partial = f"{os.path.splitext(csv_filename)[0]}.partial.csv"
        df_results.to_csv(partial, index=False)
        sys.exit(
            f"{len(errors)} dataset(s) failed; {csv_filename} was not written. "
            f"The {len(rows)} successful rows are in {partial}."
        )

    df_results.to_csv(csv_filename, index=False)
    print(f"Saved results to {csv_filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="Number of datasets benchmarked in parallel (one process per dataset).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gp_tuned", "gp_default", "dt", "boolxai", "ripper"],
        default="gp_tuned",
        help=(
            "Model to run: 'gp_tuned' (default), 'gp_default', 'dt' (Decision Tree), "
            "'boolxai', or 'ripper' (RIPPER from wittgenstein, default parameters)."
        ),
    )
    parser.add_argument("--n_runs", type=int, default=5, help="Number of runs.")
    parser.add_argument(
        "--n_folds", type=int, default=3, help="Number of cross-validation folds."
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=BASE_SEED,
        help="Run i uses seed base_seed + i for its splits, for every model.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory for storing datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to run. Default: every binary classification dataset in PMLB.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path of the result CSV. Default: a name derived from the model options.",
    )
    parser.add_argument(
        "--max_leaf_nodes",
        type=int,
        default=None,
        help="Maximum number of leaf nodes for Decision Tree.",
    )
    parser.add_argument(
        "--binarizer",
        type=str,
        choices=["none", "standard", "sklearn"],
        default=None,
        help=(
            "Binarizer for boolxai and ripper. 'standard' is hgp-lib's "
            "StandardBinarizer(num_bins=5); 'sklearn' is a 5-bin quantile "
            "KBinsDiscretizer; 'none' passes the raw data. "
            "Default: 'standard' for boolxai, 'none' for ripper."
        ),
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of epochs for GP default benchmark.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=0,
        help="Depth of the population hierarchy for gp_default (0 = a single population).",
    )
    parser.add_argument(
        "--num_child_populations",
        type=int,
        default=3,
        help="Child populations per parent for gp_default when max_depth > 0.",
    )
    parser.add_argument(
        "--feature_fraction",
        type=float,
        default=0.33,
        help="Fraction of features given to each child population when max_depth > 0.",
    )
    args = parser.parse_args()

    if args.binarizer is None:
        args.binarizer = "none" if args.model == "ripper" else "standard"
    if args.model == "boolxai" and args.binarizer == "none":
        parser.error("boolxai needs binary input: use --binarizer standard or sklearn.")
    if args.model == "ripper" and args.binarizer == "sklearn":
        parser.error("ripper supports --binarizer none or standard.")

    dataset_names = args.datasets or get_binary_classification_datasets()

    if args.model == "gp_tuned":
        commands = get_commands_for_datasets(
            dataset_names, args.n_runs, args.n_folds, args.data_dir
        )
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            executor.map(subprocess_runner, commands)
    else:
        run_benchmarks(args, dataset_names)


if __name__ == "__main__":
    freeze_support()
    main()
