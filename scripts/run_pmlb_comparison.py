"""
Run the PMLB comparison of hgp-lib against DT, BoolXAI, and RIPPER, then summarize it.

Each experiment calls ``scripts/run_on_pmlb.py``, which benchmarks one method on every
dataset with one process per dataset. An experiment whose CSV already exists in the
output folder is skipped, so running this script again only runs what is missing.
At the end, the script prints the mean rank, the number of datasets where each method
is best or tied for best, the mean test F1, the mean number of literals and nodes of the
learned rules, and the mean runtime per dataset, and saves them to ``summary.md``. The table of each HGP variant against the baselines also shows
the HGP win rate: the share of datasets on which that variant scores strictly higher
than each baseline. A separate table compares the HGP variants with each other. It also
saves radar plots (``plot_radar.py``) of all methods, of the HGP variants together, of
each HGP variant alone, and of each HGP variant against the baselines.

BoolXAI pins numpy 1.26.4 and scikit-learn 1.5.2, so it runs in its own Python 3.10 or
3.11 venv (``--boolxai_python``). Every other experiment runs with the interpreter that
runs this script.

Usage:
    python scripts/run_pmlb_comparison.py
    python scripts/run_pmlb_comparison.py --test_run   # breast_cancer_50 only, no CSVs
"""

import argparse
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from plot_radar import is_tie, plot_radar_performance
from prettytable import PrettyTable, TableStyle

matplotlib.use("Agg")  # Save the plots without opening a window.

ROOT = Path(__file__).resolve().parents[1]
RUN_ON_PMLB = ROOT / "scripts" / "run_on_pmlb.py"
# 50 stratified rows of breast_cancer, for --test_run.
TEST_RUN_DATASET = "breast_cancer_50"


@dataclass(frozen=True)
class Experiment:
    label: str
    csv_name: str
    args: tuple[str, ...]
    color: str
    is_hgp: bool = False
    needs_boolxai_python: bool = False


def _hgp(epochs: int, depth: int, color: str) -> Experiment:
    return Experiment(
        label=f"HGP ({epochs} epochs)"
        if depth == 0
        else f"HGP ({epochs} epochs, depth {depth})",
        csv_name=f"pmlb_hgp_epochs{epochs}_depth{depth}.csv",
        args=(
            "--model",
            "gp_default",
            "--num_epochs",
            str(epochs),
            "--max_depth",
            str(depth),
        ),
        color=color,
        is_hgp=True,
    )


# Run in this order.
EXPERIMENTS = (
    _hgp(1000, 0, "royalblue"),
    # _hgp(1000, 1, "dodgerblue"),
    _hgp(10000, 0, "navy"),
    # _hgp(10000, 1, "teal"),
    Experiment("DT", "pmlb_dt.csv", ("--model", "dt"), color="crimson"),
    Experiment(
        "BoolXAI",
        "pmlb_boolxai_sklearn.csv",
        ("--model", "boolxai", "--binarizer", "sklearn"),
        color="darkorange",
        needs_boolxai_python=True,
    ),
    Experiment(
        "RIPPER",
        "pmlb_ripper_none.csv",
        ("--model", "ripper", "--binarizer", "none"),
        color="forestgreen",
    ),
    # Experiment(
    #     "RIPPER (hgp-lib bins)",
    #     "pmlb_ripper_standard.csv",
    #     ("--model", "ripper", "--binarizer", "standard"),
    #     color="purple",
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


HGP_WIN_RATE = "HGP win rate"
# Table columns that average a per-dataset CSV column over the datasets. The rule
# columns count the literals and the nodes (literals and operators) of the rule learned
# in each fold, already averaged over folds and runs in the CSV.
AVERAGED_COLUMNS = {
    "Mean literals": "mean_n_literals",
    "Mean nodes": "mean_rule_size",
    "Mean runtime (s)": "runtime_s",
}


def summarize(
    scores: pd.DataFrame, averaged: dict[str, pd.DataFrame], hgp: str | None = None
) -> pd.DataFrame:
    """
    Mean rank, best-or-tied count, and mean F1 of every method (column) over the
    datasets (rows), and the mean of every frame in ``averaged`` over the same datasets.
    Rank 1 is the best score on a dataset; tied methods share the average of their
    ranks. Ties use ``plot_radar.is_tie``.

    With ``hgp``, also the HGP win rate: the percentage of datasets on which that HGP
    variant scores strictly higher than each other method. Ties are not wins.
    """
    values = scores.to_numpy()
    a, b = values[:, :, None], values[:, None, :]
    close = is_tie(a, b)
    better = (b > a) & ~close  # [d, i, j]: method j beats method i on dataset d
    ranks = 1 + better.sum(axis=2) + (close.sum(axis=2) - 1) / 2
    best = is_tie(values, values.max(axis=1, keepdims=True))
    table = pd.DataFrame(
        {
            "Mean rank": ranks.mean(axis=0),
            "Best or tied": best.sum(axis=0),
            "Mean F1": values.mean(axis=0),
            **{
                name: frame.loc[scores.index, scores.columns].mean(axis=0).to_numpy()
                for name, frame in averaged.items()
            },
        },
        index=pd.Index(scores.columns, name="Method"),
    )
    if hgp is not None:
        hgp_scores = scores[hgp].to_numpy()
        win_rates = {
            method: 100
            * ((hgp_scores > values[:, i]) & ~is_tie(hgp_scores, values[:, i])).mean()
            for i, method in enumerate(scores.columns)
        }
        table[HGP_WIN_RATE] = [
            "-" if method == hgp else f"{win_rates[method]:.2f}%"
            for method in scores.columns
        ]
    return table.sort_values("Mean rank")


def make_table(table: pd.DataFrame, style: TableStyle) -> PrettyTable:
    pretty = PrettyTable([table.index.name, *table.columns])
    pretty.set_style(style)
    for method, row in table.iterrows():
        cells = [
            method,
            f"{row['Mean rank']:.2f}",
            int(row["Best or tied"]),
            f"{row['Mean F1']:.4f}",
            *("-" if pd.isna(row[c]) else f"{row[c]:.1f}" for c in AVERAGED_COLUMNS),
        ]
        if HGP_WIN_RATE in table:
            cells.append(row[HGP_WIN_RATE])
        pretty.add_row(cells)
    pretty.align = "r"
    pretty.align[table.index.name] = "l"
    return pretty


def build_summary(csv_paths: dict[Experiment, Path]) -> tuple[str, str]:
    """
    Build the comparison tables from the experiments whose CSVs exist.

    Returns:
        tuple[str, str]: The tables for the console and as Markdown.
    """
    available = {e: p for e, p in csv_paths.items() if p.is_file()}
    if not available:
        message = "No result CSVs found; nothing to summarize."
        return message, message

    frames = {
        e.label: pd.read_csv(p).set_index("dataset") for e, p in available.items()
    }
    scores = pd.DataFrame({m: f["mean_test_score"] for m, f in frames.items()})
    # Older CSVs may lack a column; the table shows "-" for it.
    averaged = {
        name: pd.DataFrame(
            {m: f.get(column, pd.Series(np.nan, f.index)) for m, f in frames.items()}
        )
        for name, column in AVERAGED_COLUMNS.items()
    }

    notes = []
    incomplete = scores.index[scores.isna().any(axis=1)]
    if len(incomplete):
        notes.append(
            f"Leaving out {len(incomplete)} datasets that some methods lack: "
            f"{', '.join(incomplete)}"
        )
    scores = scores.dropna()
    missing = [e.label for e in csv_paths if e not in available]
    if missing:
        notes.append(f"No results yet for: {', '.join(missing)}")

    sections = [("All methods", summarize(scores, averaged))]
    hgp_versions = [e.label for e in available if e.is_hgp]
    if len(hgp_versions) > 1:
        sections.append(("HGP versions", summarize(scores[hgp_versions], averaged)))
    baselines = [e.label for e in available if not e.is_hgp]
    for experiment in available:
        if experiment.is_hgp and baselines:
            methods = [experiment.label, *baselines]
            sections.append(
                (
                    f"{experiment.label} vs baselines",
                    summarize(scores[methods], averaged, hgp=experiment.label),
                )
            )

    console, markdown = list(notes), list(notes)
    for title, table in sections:
        title = f"{title} ({len(scores)} datasets)"
        pretty = make_table(table, TableStyle.SINGLE_BORDER)
        pretty.title = title
        console.append(pretty.get_string())
        markdown.append(f"## {title}\n\n{make_table(table, TableStyle.MARKDOWN)}")
    return "\n\n".join(console), "\n\n".join(markdown)


def make_plots(csv_paths: dict[Experiment, Path], out_dir: Path) -> list[Path]:
    """
    Save radar plots of all methods, of the HGP variants together, of each HGP variant
    alone, and of each HGP variant against the baselines, for the experiments whose
    CSVs exist.
    """
    available = [e for e, p in csv_paths.items() if p.is_file()]
    hgp_versions = [e for e in available if e.is_hgp]
    baselines = [e for e in available if not e.is_hgp]
    plots = [("radar_all_methods.png", available)]
    if len(hgp_versions) > 1:
        plots.append(("radar_hgp_versions.png", hgp_versions))
    for experiment in hgp_versions:
        name = experiment.csv_name.removeprefix("pmlb_").removesuffix(".csv")
        plots.append((f"radar_{name}.png", [experiment]))
        if baselines:
            plots.append((f"radar_{name}_vs_baselines.png", [experiment, *baselines]))

    saved = []
    for file_name, experiments in plots:
        if not experiments:
            continue
        series = [(str(csv_paths[e]), e.color, e.label) for e in experiments]
        path = out_dir / file_name
        # plot_radar prints its own summary, which the tables already cover.
        with contextlib.redirect_stdout(io.StringIO()):
            plot_radar_performance(series, str(path))
        saved.append(path)
    return saved


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
        default=ROOT / "results" / "pmlb",
        help="Folder for the result CSVs, summary.md, and the radar plots.",
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

        console, markdown = build_summary(csv_paths)
        plots = make_plots(csv_paths, out_dir)

    print("\n" + console)
    if not args.test_run:
        summary_path = args.out_dir / "summary.md"
        summary_path.write_text(f"# PMLB comparison\n\n{markdown}\n")
        print(f"Saved the tables to {summary_path}")
        print(f"Saved {len(plots)} plots: {', '.join(p.name for p in plots)}")
    if failed:
        sys.exit(f"Failed experiments: {', '.join(failed)}")


if __name__ == "__main__":
    main()
