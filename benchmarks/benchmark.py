import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"
RESULTS_DIR = ROOT / "benchmark-results"


def parse_args():
    parser = argparse.ArgumentParser()

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--fast", action="store_true")
    selection.add_argument("--slow", action="store_true")
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--scenario")

    parser.add_argument(
        "--name",
        default="current",
        help="Name used for the output file.",
    )
    parser.add_argument(
        "--compare",
        help="Saved pytest-benchmark run to compare against.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "pytest",
        str(BENCHMARK_DIR),
        "-o",
        "addopts=",
    ]

    if args.fast:
        command += ["-m", "fast"]
        suite = "fast"
    elif args.slow:
        command += ["-m", "slow"]
        suite = "slow"
    elif args.scenario:
        command += ["-k", args.scenario]
        suite = args.scenario
    else:
        suite = "all"

    output = RESULTS_DIR / f"{args.name}-{suite}.json"

    command += [
        f"--benchmark-json={output}",
        f"--benchmark-save={args.name}-{suite}",
    ]

    if args.compare:
        command.append(f"--benchmark-compare={args.compare}")

    print("Running:", " ".join(map(str, command)))

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()