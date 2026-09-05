import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"
RESULTS_DIR = ROOT / "benchmark-results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--fast", action="store_true")
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--scenario")

    parser.add_argument(
        "--save",
        default="current",
        help="Result label, for example 2.1.0 or 2.1.0-new-selection.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "pytest",
        str(BENCHMARK_DIR),
        "-o",
        "addopts=",
        "--benchmark-only",
    ]

    if args.fast:
        command += ["-m", "fast"]
    elif args.scenario:
        command += ["--hgp-scenario", args.scenario]

    name = args.save.removesuffix(".json")
    output = RESULTS_DIR / f"{name}.json"
    command.append(f"--benchmark-json={output}")

    print("Running:", " ".join(map(str, command)))
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Saved benchmark results to {output}")
    subprocess.run(
        [sys.executable, str(BENCHMARK_DIR / "compare.py")],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
