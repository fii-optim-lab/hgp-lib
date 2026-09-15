import argparse
import json
import re
import statistics
from collections import defaultdict
from html import escape
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def natural_key(value: str) -> tuple:
    return tuple(
        int(part.replace("_", "")) if part.replace("_", "").isdigit() else part.lower()
        for part in re.split(r"(\d[\d_]*)", value)
    )


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.2f} s"


def format_result(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def format_change(duration: float, previous: float) -> str:
    change = 100 * (duration / previous - 1)
    if abs(change) < 0.05:
        return "+0.0%"
    if change < 0:
        return f"{change:+.1f}% faster"
    return f"{change:+.1f}% slower"


def aggregate_scenarios(payload: dict) -> list[dict]:
    scenarios = defaultdict(list)
    for benchmark in payload["benchmarks"]:
        extra_info = benchmark.get("extra_info", {})
        scenario_id = extra_info.get("scenario_id")
        if scenario_id is not None:
            scenarios[scenario_id].append(benchmark)

    rows = []
    for scenario_id, benchmarks in scenarios.items():
        duration = sum(float(benchmark["stats"]["median"]) for benchmark in benchmarks)
        scored_benchmarks = [
            benchmark
            for benchmark in benchmarks
            if benchmark.get("extra_info", {}).get("test_score") is not None
        ]
        scores = [
            float(benchmark["extra_info"]["test_score"])
            for benchmark in scored_benchmarks
        ]

        score_mean = statistics.mean(scores) if scores else None
        if len(scores) > 1:
            score_std = statistics.pstdev(scores)
        elif scores:
            score_std = scored_benchmarks[0]["extra_info"].get("test_score_std")
        else:
            score_std = None

        rows.append(
            {
                "scenario": scenario_id,
                "duration": duration,
                "score_mean": score_mean,
                "score_std": None if score_std is None else float(score_std),
            }
        )
    return rows


def load_results() -> list[dict]:
    rows = []
    for path in sorted(
        RESULTS_DIR.glob("*.json"), key=lambda item: natural_key(item.stem)
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload["hgp_benchmark"]
        for row in aggregate_scenarios(payload):
            row.update(
                {
                    "machine": metadata["machine"],
                    "version": metadata["version"],
                    "name": metadata.get("name"),
                }
            )
            rows.append(row)
    return rows


def select_machines(rows: list[dict], selected: list[str] | None) -> list[str]:
    available = sorted({row["machine"] for row in rows}, key=natural_key)
    if selected is None:
        return available

    machines = []
    for machine in selected:
        if machine not in available:
            raise ValueError(f"Unknown machine: {machine}")
        if machine not in machines:
            machines.append(machine)
    return machines


def has_distinct_results(
    rows: list[dict], machine: str, reference_machine: str
) -> bool:
    reference = {
        row["version"]: format_result(row["score_mean"], row["score_std"])
        for row in rows
        if row["machine"] == reference_machine and row["score_mean"] is not None
    }
    for row in rows:
        if row["machine"] != machine or row["score_mean"] is None:
            continue
        result = format_result(row["score_mean"], row["score_std"])
        if reference.get(row["version"]) != result:
            return True
    return False


def build_table(rows: list[dict], machines: list[str]) -> list[str]:
    first_machine = machines[0]
    result_columns = {
        machine: (
            any(
                row["machine"] == machine and row["score_mean"] is not None
                for row in rows
            )
            if machine == first_machine
            else has_distinct_results(rows, machine, first_machine)
        )
        for machine in machines
    }
    columns_per_machine = {
        machine: 2 + int(result_columns[machine]) for machine in machines
    }

    lines = ["<table>", "  <thead>", "    <tr>"]
    lines.append('      <th rowspan="2">Version</th>')
    for machine in machines:
        lines.append(
            f'      <th colspan="{columns_per_machine[machine]}">{escape(machine)}</th>'
        )
    lines.extend(["    </tr>", "    <tr>"])
    for machine in machines:
        lines.extend(["      <th>Time</th>", "      <th>vs previous</th>"])
        if result_columns[machine]:
            lines.append("      <th>Result</th>")
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])

    lookup = {(row["machine"], row["version"]): row for row in rows}
    versions = sorted({row["version"] for row in rows}, key=natural_key)
    previous_durations = {}
    for version in versions:
        lines.extend(["    <tr>", f"      <td>{escape(version)}</td>"])
        for machine in machines:
            row = lookup.get((machine, version))
            if row is None:
                lines.extend(
                    "      <td>-</td>" for _ in range(columns_per_machine[machine])
                )
                continue

            previous = previous_durations.get(machine)
            change = (
                "-" if previous is None else format_change(row["duration"], previous)
            )
            lines.extend(
                [
                    f'      <td align="right">{format_duration(row["duration"])}</td>',
                    f'      <td align="right">{change}</td>',
                ]
            )
            if result_columns[machine]:
                result = format_result(row["score_mean"], row["score_std"])
                lines.append(f'      <td align="right">{result}</td>')
            previous_durations[machine] = row["duration"]
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return lines


def build_report(rows: list[dict], selected_machines: list[str] | None = None) -> str:
    machines = select_machines(rows, selected_machines)
    filtered_rows = [row for row in rows if row["machine"] in machines]
    groups = defaultdict(list)
    for row in filtered_rows:
        groups[(row["scenario"], row["name"])].append(row)

    lines = ["# hgp-lib performance report", ""]
    group_keys = sorted(
        groups,
        key=lambda key: (natural_key(key[0]), natural_key(key[1] or "")),
    )
    for scenario, name in group_keys:
        heading = f"## {scenario}"
        if name:
            heading += f" - {name}"
        lines.extend([heading, ""])
        group_rows = groups[(scenario, name)]
        group_machines = [
            machine
            for machine in machines
            if any(row["machine"] == machine for row in group_rows)
        ]
        lines.extend(build_table(group_rows, group_machines))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--machine",
        action="append",
        dest="machines",
        help="Machine to include. Repeat to include multiple machines.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_results()
    if not rows:
        raise SystemExit(f"No benchmark results found in {RESULTS_DIR}")

    try:
        report = build_report(rows, args.machines)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.output is None:
        print(report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
