import argparse
import json
import re
import statistics
from collections import defaultdict
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
        return "—"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def format_change(duration: float, baseline: float) -> str:
    change = 100 * (duration / baseline - 1)
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
        payload = json.loads(path.read_text())
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


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


def build_report(rows: list[dict]) -> str:
    lines = ["# HGP performance report", ""]
    machines = sorted({row["machine"] for row in rows}, key=natural_key)

    for machine in machines:
        lines.extend([f"## Machine: {markdown_escape(machine)}", ""])
        machine_rows = [row for row in rows if row["machine"] == machine]
        groups = defaultdict(list)
        for row in machine_rows:
            groups[(row["scenario"], row["name"])].append(row)

        group_keys = sorted(
            groups,
            key=lambda key: (natural_key(key[0]), natural_key(key[1] or "")),
        )
        for scenario, name in group_keys:
            heading = f"### {markdown_escape(scenario)}"
            if name:
                heading += f" — {markdown_escape(name)}"
            lines.extend(
                [
                    heading,
                    "",
                    "| Version | Time | vs first | Result |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )

            versions = sorted(
                groups[(scenario, name)], key=lambda row: natural_key(row["version"])
            )
            baseline = versions[0]["duration"]
            for index, row in enumerate(versions):
                change = "—" if index == 0 else format_change(row["duration"], baseline)
                lines.append(
                    f"| {markdown_escape(row['version'])} "
                    f"| {format_duration(row['duration'])} "
                    f"| {change} "
                    f"| {format_result(row['score_mean'], row['score_std'])} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_results()
    if not rows:
        raise SystemExit(f"No benchmark results found in {RESULTS_DIR}")

    report = build_report(rows)
    if args.output is None:
        print(report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
