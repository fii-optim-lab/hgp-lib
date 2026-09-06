import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"
RESULTS_DIR = BENCHMARK_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--fast", action="store_true")
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--scenario")

    parser.add_argument("--machine", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--name")

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

    parts = [args.machine, args.version]
    if args.name:
        parts.append(args.name)
    output = RESULTS_DIR / f"{'-'.join(parts)}.json"
    command.append(f"--benchmark-json={output}")

    print("Running:", " ".join(map(str, command)))
    subprocess.run(command, cwd=ROOT, check=True)

    payload = json.loads(output.read_text())
    payload["hgp_benchmark"] = {
        "machine": args.machine,
        "version": args.version,
        "name": args.name,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved benchmark results to {output}")


if __name__ == "__main__":
    main()
