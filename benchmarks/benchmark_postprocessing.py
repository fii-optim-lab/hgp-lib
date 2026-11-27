"""
Time `hgp_lib.postprocessing` on evolved rules and on random rules.

The evolved rules are those of ``benchmarks/artifacts/rule_evaluation``: each of the 25
artifacts holds the 100 rules of a population evolved on one fold of a dataset (see
``benchmarks/README.md``). The random rules have a fixed number of nodes and use the
features of the breast cancer fold 0; they have none of the repeated parts of evolved
rules, so optimizing removes little and the normal forms are large.

For every rule, this script times ``postprocess`` with ``optimize=True`` and with
``canonical="dnf"`` and ``"cnf"``, checks that each result predicts the same as the rule
on the fold's test rows, and prints a Markdown report grouped by rule size:

    python benchmarks/benchmark_postprocessing.py [--time-limit 10] [--output report.md]

A call that runs longer than ``--time-limit`` seconds is stopped and counted as such
(this uses ``SIGALRM``, so it needs a Unix system).
"""

import argparse
import signal
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from hgp_lib.evaluation import predict
from hgp_lib.postprocessing import postprocess
from hgp_lib.preprocessing import StandardBinarizer
from hgp_lib.rules import And, Literal, Or

ROOT = Path(__file__).resolve().parents[1]

BUCKETS = [(1, 25), (26, 50), (51, 100), (101, 250), (251, 500)]
MODES = {"optimize": None, "dnf": "dnf", "cnf": "cnf"}
RANDOM_SIZES = (10, 15, 20, 25, 30, 40, 50, 100)
RANDOM_RULES = 50


class TimeLimit(Exception):
    pass


def _alarm(signum, frame):
    raise TimeLimit


def run(rule, binarizer, canonical, limit: float):
    """The result and the duration of one call, or ``(None, None)`` over the limit."""
    signal.setitimer(signal.ITIMER_REAL, limit)
    start = time.perf_counter()
    try:
        result = postprocess(rule, binarizer, canonical=canonical)
    except TimeLimit:
        return None, None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return result, time.perf_counter() - start


def fitted_fold(dataset: str, fold: int):
    """The binarizer fitted on the fold's training rows, and its test rows."""
    from benchmarks.data import load_fold

    X_train, X_test, y_train, _ = load_fold(dataset, fold)
    binarizer = StandardBinarizer(progress_bar=False)
    binarizer.fit_transform(X_train, y_train)
    return binarizer, binarizer.transform(X_test).to_numpy()


def measure_rule(rule, binarizer, data, limit: float) -> dict:
    expected = predict(rule, data)
    record = {"size": len(rule)}
    for mode, canonical in MODES.items():
        result, seconds = run(rule, binarizer, canonical, limit)
        if result is not None and not np.array_equal(predict(result, data), expected):
            raise AssertionError(f"{mode} predicts differently for {rule}")
        record[mode] = seconds
        record[f"{mode}_size"] = None if result is None else len(result)
    return record


def measure_evolved(limit: float) -> list[dict]:
    from benchmarks.data import DATASET_NAMES, N_SPLITS
    from benchmarks.rule_artifacts import load_artifact

    records = []
    for dataset in DATASET_NAMES:
        for fold in range(N_SPLITS):
            rules, feature_mapping, _ = load_artifact(dataset, fold)
            binarizer, data = fitted_fold(dataset, fold)
            names = binarizer.get_feature_names_out()
            if dict(enumerate(names)) != feature_mapping:
                raise ValueError(f"{dataset} fold {fold}: the features differ")
            records.extend(measure_rule(rule, binarizer, data, limit) for rule in rules)
            print(f"{dataset} fold {fold}: done", file=sys.stderr)
    return records


def random_rule(rng: np.random.Generator, size: int, num_features: int):
    """A random rule of exactly ``size`` nodes (``size`` is 1 or at least 3)."""
    if size == 1:
        return Literal(
            value=int(rng.integers(num_features)), negated=bool(rng.random() < 0.5)
        )
    # Split the size - 1 other nodes between 2 to 4 children, none of size 2.
    while True:
        count = int(rng.integers(2, min(4, size - 1) + 1))
        cuts = np.sort(rng.choice(np.arange(1, size - 1), count - 1, replace=False))
        sizes = np.diff([0, *cuts, size - 1])
        if 2 not in sizes:
            break
    operator = And if rng.random() < 0.5 else Or
    children = [random_rule(rng, int(n), num_features) for n in sizes]
    return operator(children, negated=bool(rng.random() < 0.25), copy_subrules=False)


def measure_random(limit: float) -> list[dict]:
    binarizer, data = fitted_fold("breast_cancer", 0)
    num_features = len(binarizer.get_feature_names_out())
    rng = np.random.default_rng(0)
    records = []
    for size in RANDOM_SIZES:
        for _ in range(RANDOM_RULES):
            rule = random_rule(rng, size, num_features)
            records.append(measure_rule(rule, binarizer, data, limit))
        print(f"random rules of {size} nodes: done", file=sys.stderr)
    return records


def _seconds(value: float, limit: float) -> str:
    return f"> {limit:g}" if value == float("inf") else f"{value:.3f}"


def report(records: list[dict], buckets: list, limit: float) -> str:
    """
    One row per size bucket. The times are in seconds; a call stopped at the time limit
    counts as longer than the limit.
    """
    lines = [
        "| Rule size | Rules | Median size after optimize | "
        + " | ".join(f"{mode}: median | {mode}: 90% | {mode}: max" for mode in MODES)
        + " |",
        "|" + "---:|" * (3 + 3 * len(MODES)),
    ]
    for low, high in buckets:
        bucket = [r for r in records if low <= r["size"] <= high]
        if not bucket:
            continue
        cells = [str(low) if low == high else f"{low}-{high}", str(len(bucket))]
        cells.append(f"{statistics.median(r['optimize_size'] for r in bucket):g}")
        for mode in MODES:
            times = sorted(float("inf") if r[mode] is None else r[mode] for r in bucket)
            ninetieth = times[min(len(times) - 1, int(0.9 * len(times)))]
            for value in (statistics.median(times), ninetieth, times[-1]):
                cells.append(_seconds(value, limit))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    signal.signal(signal.SIGALRM, _alarm)
    evolved = report(measure_evolved(args.time_limit), BUCKETS, args.time_limit)
    random_buckets = [(size, size) for size in RANDOM_SIZES]
    random = report(measure_random(args.time_limit), random_buckets, args.time_limit)
    text = f"## Evolved rules\n\n{evolved}\n\n## Random rules\n\n{random}"
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
