"""
Run the PMLB comparison of hgp-lib against DT, BoolXAI, and RIPPER, then summarize it.

Each experiment calls ``scripts/run_on_pmlb.py``, which benchmarks one method on every
dataset with one process per dataset. An experiment whose CSV already exists in the
output folder is skipped, so running this script again only runs what is missing.
At the end, the script prints the mean rank, the number of datasets where each method
is best or tied for best, and the mean test F1, and saves them to ``summary.md``.

BoolXAI pins numpy 1.26.4 and scikit-learn 1.5.2, so it runs in its own Python 3.10 or
3.11 venv (``--boolxai_python``). Every other experiment runs with the interpreter that
runs this script.

Usage:
    python scripts/run_pmlb_comparison.py
    python scripts/run_pmlb_comparison.py --test_run   # breast_cancer_50 only, no CSVs
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN_ON_PMLB = ROOT / "scripts" / "run_on_pmlb.py"
TIE_RTOL = 1e-5
TIE_ATOL = 1e-8
# 50 stratified rows of breast_cancer, for --test_run.
TEST_RUN_DATASET = "breast_cancer_50"


@dataclass(frozen=True)
class Experiment:
    label: str
    csv_name: str
    args: tuple[str, ...]
    is_hgp: bool = False
    needs_boolxai_python: bool = False


def _hgp(epochs: int, depth: int) -> Experiment:
    return Experiment(
        label=f"HGP ({epochs} epochs, depth {depth})",
        csv_name=f"pmlb_hgp_epochs{epochs}_depth{depth}.csv",
        args=(
            "--model",
            "gp_default",
            "--num_epochs",
            str(epochs),
            "--max_depth",
            str(depth),
        ),
        is_hgp=True,
    )


# Run in this order.
EXPERIMENTS = (
    # _hgp(1000, 0),
    # _hgp(1000, 1),
    # _hgp(10000, 0),
    _hgp(10000, 1),
    # Experiment("DT", "pmlb_dt.csv", ("--model", "dt")),
    # Experiment(
    #     "BoolXAI (sklearn bins)",
    #     "pmlb_boolxai_sklearn.csv",
    #     ("--model", "boolxai", "--binarizer", "sklearn"),
    #     needs_boolxai_python=True,
    # ),
    # Experiment(
    #     "RIPPER (raw)",
    #     "pmlb_ripper_none.csv",
    #     ("--model", "ripper", "--binarizer", "none"),
    # ),
    # Experiment(
    #     "RIPPER (hgp-lib bins)",
    #     "pmlb_ripper_standard.csv",
    #     ("--model", "ripper", "--binarizer", "standard"),
    # ),
)


def resolve_datasets(data_dir: Path, test_run: bool) -> list[str]:
    """Return the datasets to run, downloading the ones that are not cached yet."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from preprocess.pmlb_preprocess import save_pmlb_data
    from run_on_pmlb import get_binary_classification_datasets

    names = [TEST_RUN_DATASET] if test_run else get_binary_classification_datasets()
    data_dir.mkdir(parents=True, exist_ok=True)
    available = []
    unavailable = []
    for name in names:
        if not (data_dir / f"{name}.hdf").is_file():
            try:
                save_pmlb_data(name, str(data_dir))
            except Exception:  # noqa: BLE001
                unavailable.append(name)
                continue
        available.append(name)
    if unavailable:
        print(
            f"Skipping {len(unavailable)} datasets listed in the PMLB summary that "
            f"PMLB cannot download: {', '.join(unavailable)}"
        )
    print(f"Running on {len(available)} datasets.")
    return available


def run_experiment(
    experiment: Experiment,
    csv_path: Path,
    datasets: list[str],
    python: str,
    args: argparse.Namespace,
) -> bool:
    command = [
        python,
        str(RUN_ON_PMLB),
        *experiment.args,
        "--n_jobs",
        str(args.n_jobs),
        "--n_runs",
        str(args.n_runs),
        "--n_folds",
        str(args.n_folds),
        "--base_seed",
        str(args.base_seed),
        "--data_dir",
        str(args.data_dir),
        "--output",
        str(csv_path),
        "--datasets",
        *datasets,
    ]
    # Parallelism is across datasets, so keep every worker single-threaded.
    env = {
        **os.environ,
        # "OMP_NUM_THREADS": "1",
        # "OPENBLAS_NUM_THREADS": "1",
        # "MKL_NUM_THREADS": "1",
        # "VECLIB_MAXIMUM_THREADS": "1",
        # "NUMEXPR_NUM_THREADS": "1",
    }
    print(f"\n=== {experiment.label}: running with {python}")
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        print(f"=== {experiment.label}: FAILED after {elapsed:.0f}s")
        return False
    print(f"=== {experiment.label}: done in {elapsed:.0f}s")
    return True


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Mean rank, best-or-tied count, and mean F1 of every method (column) over the
    datasets (rows). Rank 1 is the best score on a dataset; tied methods share the
    average of their ranks. Two scores tie when they are equal within the tolerance
    used by ``plot_radar.py``.
    """
    values = scores.to_numpy()
    a, b = values[:, :, None], values[:, None, :]
    close = np.isclose(a, b, rtol=TIE_RTOL, atol=TIE_ATOL) | np.isclose(
        b, a, rtol=TIE_RTOL, atol=TIE_ATOL
    )
    better = (b > a) & ~close  # [d, i, j]: method j beats method i on dataset d
    ranks = 1 + better.sum(axis=2) + (close.sum(axis=2) - 1) / 2
    best = np.isclose(
        values, values.max(axis=1, keepdims=True), rtol=TIE_RTOL, atol=TIE_ATOL
    )
    table = pd.DataFrame(
        {
            "Mean rank": ranks.mean(axis=0),
            "Best or tied": best.sum(axis=0),
            "Mean F1": values.mean(axis=0),
        },
        index=pd.Index(scores.columns, name="Method"),
    )
    return table.sort_values("Mean rank")


def format_table(table: pd.DataFrame) -> list[str]:
    header = f"| {table.index.name} | " + " | ".join(table.columns) + " |"
    lines = [header, "|" + "---|" * (len(table.columns) + 1)]
    for method, row in table.iterrows():
        lines.append(
            f"| {method} | {row['Mean rank']:.2f} | {int(row['Best or tied'])} "
            f"| {row['Mean F1']:.4f} |"
        )
    return lines


def build_summary(csv_paths: dict[Experiment, Path]) -> list[str]:
    """Build the comparison tables from the experiments whose CSVs exist."""
    available = {e: p for e, p in csv_paths.items() if p.is_file()}
    if not available:
        return ["No result CSVs found; nothing to summarize."]

    scores = pd.DataFrame(
        {
            e.label: pd.read_csv(p).set_index("dataset")["mean_test_score"]
            for e, p in available.items()
        }
    )
    lines = []
    incomplete = scores.index[scores.isna().any(axis=1)]
    if len(incomplete):
        lines.append(
            f"Leaving out {len(incomplete)} datasets that some methods lack: "
            f"{', '.join(incomplete)}"
        )
        lines.append("")
    scores = scores.dropna()
    missing = [e.label for e in csv_paths if e not in available]
    if missing:
        lines.append(f"No results yet for: {', '.join(missing)}")
        lines.append("")

    groups = [("All methods", list(scores.columns))]
    baselines = [e.label for e in available if not e.is_hgp]
    for experiment in available:
        if experiment.is_hgp and baselines:
            groups.append(
                (f"{experiment.label} vs baselines", [experiment.label, *baselines])
            )

    for title, methods in groups:
        lines.append(f"## {title} ({len(scores)} datasets)")
        lines.append("")
        lines.extend(format_table(summarize(scores[methods])))
        lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Run the PMLB comparison experiments and summarize them."
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help=f"Run every experiment on {TEST_RUN_DATASET} only and print the tables "
        "without writing any CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=ROOT / "softwareX" / "new_results",
        help="Folder for the result CSVs and summary.md.",
    )
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--boolxai_python",
        type=str,
        default=str(ROOT / ".venv" / "boolxai" / "bin" / "python"),
        help="Python of the venv with boolxai, numpy 1.26.4, and scikit-learn 1.5.2.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Datasets run in parallel. Default: all cores.",
    )
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--base_seed", type=int, default=0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) if args.test_run else args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_paths = {e: out_dir / e.csv_name for e in EXPERIMENTS}

        datasets = None
        failed = []
        for experiment, csv_path in csv_paths.items():
            if csv_path.is_file():
                print(f"Skipping {experiment.label}: {csv_path} exists.")
                continue
            python = sys.executable
            if experiment.needs_boolxai_python:
                python = args.boolxai_python
                if not Path(python).is_file():
                    print(f"=== {experiment.label}: no Python found at {python}")
                    failed.append(experiment.label)
                    continue
            if datasets is None:
                datasets = resolve_datasets(args.data_dir, args.test_run)
            if not run_experiment(experiment, csv_path, datasets, python, args):
                failed.append(experiment.label)

        lines = build_summary(csv_paths)

    print("\n" + "\n".join(lines))
    if not args.test_run:
        summary_path = args.out_dir / "summary.md"
        summary_path.write_text("# PMLB comparison\n\n" + "\n".join(lines))
        print(f"Saved the tables to {summary_path}")
    if failed:
        sys.exit(f"Failed experiments: {', '.join(failed)}")


if __name__ == "__main__":
    main()
