"""
Benchmark every option combination of every evaluation backend, on one version.

Runs the scenarios in ``benchmarks/backends`` and saves, under
``benchmarks/results/backends``:

- ``<machine>-<version>[-<name>].json``: the pytest-benchmark results.
- ``<machine>-<version>[-<name>].md``: a report ranking the combinations.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"
SUITE_DIR = BENCHMARK_DIR / "backends"
RESULTS_DIR = BENCHMARK_DIR / "results" / "backends"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--machine", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--name")
    return parser.parse_args()


def natural_key(value: str) -> tuple:
    """Sort ``1_000_rows`` before ``10_000_rows``."""
    return tuple(
        (0, int(part.replace("_", "")))
        if part.replace("_", "").isdigit()
        else (1, part)
        for part in re.split(r"(\d[\d_]*)", value)
    )


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.2f} s"


def load_rows(payload: dict) -> dict[str, list[dict]]:
    """Group the median time and result of each backend by scenario."""
    scenarios = defaultdict(list)
    for benchmark in payload["benchmarks"]:
        info = benchmark["extra_info"]
        scenarios[info["scenario_id"]].append(
            {
                "backend": info["backend"],
                "is_default": info.get("is_default", False),
                "time": float(benchmark["stats"]["median"]),
                "result": info.get("test_score"),
            }
        )
    return scenarios


def backend_name(row: dict) -> str:
    return f"{row['backend']} (default)" if row["is_default"] else row["backend"]


def build_summary(scenarios: dict[str, list[dict]]) -> list[str]:
    """Rank backends by the geometric mean of their time relative to the fastest."""
    relative = defaultdict(list)
    fastest_in = Counter()
    names = {}
    for rows in scenarios.values():
        fastest = min(row["time"] for row in rows)
        fastest_in[min(rows, key=lambda row: row["time"])["backend"]] += 1
        for row in rows:
            relative[row["backend"]].append(row["time"] / fastest)
            names[row["backend"]] = backend_name(row)

    ranking = sorted(
        relative, key=lambda backend: statistics.geometric_mean(relative[backend])
    )
    lines = [
        "| Rank | Backend | Time vs fastest (geometric mean) | Worst scenario | Fastest in |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for rank, backend in enumerate(ranking, start=1):
        lines.append(
            f"| {rank} | `{names[backend]}` "
            f"| x{statistics.geometric_mean(relative[backend]):.2f} "
            f"| x{max(relative[backend]):.2f} "
            f"| {fastest_in[backend]} / {len(scenarios)} |"
        )
    return lines


def build_scenario_table(rows: list[dict]) -> list[str]:
    rows = sorted(rows, key=lambda row: row["time"])
    fastest = rows[0]["time"]
    lines = [
        "| Backend | Time | vs fastest | Result |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        result = "-" if row["result"] is None else f"{row['result']:.6f}"
        lines.append(
            f"| `{backend_name(row)}` | {format_duration(row['time'])} "
            f"| x{row['time'] / fastest:.2f} | {result} |"
        )
    return lines


def build_report(payload: dict) -> str:
    metadata = payload["hgp_benchmark"]
    scenarios = load_rows(payload)
    title = f"# hgp-lib backend report: {metadata['machine']}, {metadata['version']}"
    if metadata.get("name"):
        title += f", {metadata['name']}"

    lines = [
        title,
        "",
        "Times are medians. Each scenario's result must be the same for every backend.",
        "",
        "## Summary",
        "",
        *build_summary(scenarios),
    ]

    differing = [
        scenario
        for scenario, rows in scenarios.items()
        if len({row["result"] for row in rows}) > 1
    ]
    if differing:
        lines.extend(["", "**Results differ between backends in:**", ""])
        lines.extend(
            f"- `{scenario}`" for scenario in sorted(differing, key=natural_key)
        )

    for scenario in sorted(scenarios, key=natural_key):
        lines.extend(
            ["", f"## {scenario}", "", *build_scenario_table(scenarios[scenario])]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    parts = [args.machine, args.version]
    if args.name:
        parts.append(args.name)
    output = RESULTS_DIR / f"{'-'.join(parts)}.json"

    command = [
        sys.executable,
        "-m",
        "pytest",
        str(SUITE_DIR),
        "-o",
        "addopts=",
        "--benchmark-only",
        f"--benchmark-json={output}",
    ]
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

    report = output.with_suffix(".md")
    report.write_text(build_report(payload), encoding="utf-8")
    print(f"Saved report to {report}")


if __name__ == "__main__":
    main()
