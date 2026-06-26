"""
Manual integration test for ``GPBenchmarker._run_parallel`` and ``ProgressListener``.

Goal:
    Exercise the full parallel pipeline (multiprocessing.Pool + Queue + tqdm
    progress bars) end-to-end, so that we can:
      * visually verify that the run / fold / epoch progress bars all advance,
      * verify that the parent process exits cleanly (no hanging feeder
        threads or leaked queue/Pool resources),
      * stress the lifecycle of ``multiprocessing.Queue`` + ``ProgressListener``
        the same way it gets stressed under ``optuna_hypertuning.py``.

The on-disk dataset is intentionally tiny: real workloads come from PMLB and
take seconds-to-minutes per fold. To get realistic timing without that, we
inject a sleep into the score function so that each fitness evaluation costs a
controllable wall-clock amount.

Run manually:

    python tests/integration_test_parallel_progress.py

This file deliberately does *not* start with ``test_`` so that pytest's default
discovery (``test_*.py`` / ``*_test.py``) will skip it. It can still be invoked
explicitly with ``pytest tests/integration_test_parallel_progress.py`` if you
want.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from dataclasses import dataclass
from multiprocessing import freeze_support

import numpy as np
import pandas as pd

from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.benchmarkers.progress import ProgressConfig, ProgressListener
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig

# Sleep injected into every fitness evaluation. With population_size=20,
# num_epochs=10 and 5 parallel runs * 2 folds, this gives the bars enough
# wall-clock time to actually be observed. Tune this if the test runs too fast
# or too slow on your machine.
DEFAULT_EVAL_SLEEP_S = 0.0005


def _slow_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Score function that mimics a non-trivial workload via ``time.sleep``.

    Defined at module level so it is picklable by ``multiprocessing.Pool``
    workers under both ``fork`` and ``spawn`` start methods.
    """
    time.sleep(DEFAULT_EVAL_SLEEP_S)
    return float(np.mean(predictions == labels))


def _make_dataset(n_rows: int = 200, n_features: int = 6, seed: int = 0):
    """Return a small but non-degenerate boolean classification dataset."""
    rng = np.random.default_rng(seed)
    matrix = rng.integers(0, 2, size=(n_rows, n_features), dtype=np.int8).astype(bool)
    df = pd.DataFrame(matrix, columns=[f"f{i}" for i in range(n_features)])
    # A trivially-learnable target so the GP doesn't degenerate, but with some
    # noise so different folds disagree.
    labels = (df["f0"] ^ df["f1"]).to_numpy().astype(int)
    flip = rng.random(n_rows) < 0.1
    labels[flip] = 1 - labels[flip]
    return df, labels


@dataclass
class CheckResult:
    ok: bool
    detail: str


def _check_progress_listener_in_isolation() -> CheckResult:
    """Sanity check: drive a ``ProgressListener`` directly with a real ``Queue``.

    This fails fast if the queue / listener wiring is broken before we even
    spin up a multiprocessing.Pool.
    """
    queue: multiprocessing.Queue = multiprocessing.Queue()
    cfg = ProgressConfig(
        total_runs=2,
        total_folds=4,
        total_epochs=10,
        show_run_progress=False,
        show_fold_progress=False,
        show_epoch_progress=False,
    )
    listener = ProgressListener(queue, cfg)
    listener.start()

    for _ in range(10):
        queue.put(("epoch", 1))
    for _ in range(4):
        queue.put(("fold", 1))
    queue.put(("run", 1))
    queue.put(("run", 1))

    listener.join()
    queue.close()
    queue.join_thread()

    alive = listener._thread is not None and listener._thread.is_alive()
    if alive:
        return CheckResult(False, "ProgressListener thread did not exit")
    return CheckResult(True, "ProgressListener exited naturally on total_runs")


def _run_parallel_benchmarker(
    n_runs: int,
    n_folds: int,
    num_epochs: int,
    population_size: int,
    show_progress: bool,
) -> CheckResult:
    """Drive a parallel ``GPBenchmarker.fit()`` and check it returns cleanly."""
    data, labels = _make_dataset()

    gp_config = BooleanGPConfig(
        score_fn=_slow_accuracy,
        optimize_scorer=False,  # keep score_fn calls direct so the sleep matters
    )
    # Override the population size via the factory used by BooleanGPConfig.
    from hgp_lib.populations import PopulationGeneratorFactory

    gp_config.population_factory = PopulationGeneratorFactory(
        population_size=population_size
    )

    trainer_config = TrainerConfig(
        gp_config=gp_config,
        num_epochs=num_epochs,
        progress_bar=show_progress,
        # Update the epoch bar every epoch so the bar moves visibly even with
        # tiny num_epochs.
        progress_update_interval=1,
    )
    config = BenchmarkerConfig(
        data=data,
        labels=labels,
        trainer_config=trainer_config,
        num_runs=n_runs,
        n_folds=n_folds,
        n_jobs=n_runs,  # one worker per run -> truly parallel
        show_run_progress=show_progress,
        show_fold_progress=show_progress,
        show_epoch_progress=show_progress,
    )

    t0 = time.monotonic()
    result = GPBenchmarker(config).fit()
    elapsed = time.monotonic() - t0

    if len(result.runs) != n_runs:
        return CheckResult(
            False,
            f"expected {n_runs} runs, got {len(result.runs)}",
        )

    seeds = [r.seed for r in result.runs]
    if len(set(seeds)) != n_runs:
        return CheckResult(False, f"runs do not have distinct seeds: {seeds}")

    for r in result.runs:
        if len(r.folds) != n_folds:
            return CheckResult(
                False,
                f"run {r.run_id} has {len(r.folds)} folds, expected {n_folds}",
            )
        if not isinstance(r.test_score, float):
            return CheckResult(
                False,
                f"run {r.run_id} test_score is not float: {type(r.test_score)}",
            )

    return CheckResult(
        True,
        f"completed {n_runs} parallel runs x {n_folds} folds x {num_epochs} epochs "
        f"in {elapsed:.2f}s (show_progress={show_progress})",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (useful for CI / non-interactive runs).",
    )
    args = parser.parse_args()

    show_progress = not args.no_progress

    print("=" * 70)
    print(" Manual integration test: parallel GPBenchmarker + ProgressListener")
    print("=" * 70)

    print("\n[1/3] ProgressListener in isolation...")
    r = _check_progress_listener_in_isolation()
    print(f"      -> {'OK' if r.ok else 'FAIL'}: {r.detail}")
    if not r.ok:
        return 1

    print(
        f"\n[2/3] Parallel benchmarker WITH progress bars "
        f"(n_runs={args.n_runs}, n_folds={args.n_folds}, "
        f"num_epochs={args.num_epochs})..."
    )
    print(
        "      Watch for three nested tqdm bars (Runs / Folds / Epochs) all advancing."
    )
    r = _run_parallel_benchmarker(
        n_runs=args.n_runs,
        n_folds=args.n_folds,
        num_epochs=args.num_epochs,
        population_size=args.population_size,
        show_progress=show_progress,
    )
    print(f"\n      -> {'OK' if r.ok else 'FAIL'}: {r.detail}")
    if not r.ok:
        return 1

    print("\n[3/3] Parallel benchmarker WITHOUT progress bars (no Queue path)...")
    r = _run_parallel_benchmarker(
        n_runs=args.n_runs,
        n_folds=args.n_folds,
        num_epochs=args.num_epochs,
        population_size=args.population_size,
        show_progress=False,
    )
    print(f"      -> {'OK' if r.ok else 'FAIL'}: {r.detail}")
    if not r.ok:
        return 1

    print("\nAll checks passed. Process should now exit cleanly.")
    return 0


if __name__ == "__main__":
    freeze_support()
    sys.exit(main())
