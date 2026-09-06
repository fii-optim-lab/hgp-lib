import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def natural_key(value: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
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
    return f"{mean:.4f} ± {std:.4f}"


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
        scores = [
            float(benchmark["extra_info"]["test_score"])
            for benchmark in benchmarks
            if benchmark.get("extra_info", {}).get("test_score") is not None
        ]

        score_mean = statistics.mean(scores) if scores else None
        if len(scores) > 1:
            score_std = statistics.pstdev(scores)
        elif scores:
            score_std = benchmarks[0].get("extra_info", {}).get("test_score_std")
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


def load_results() -> dict[str, list[dict]]:
    machines = defaultdict(list)
    for path in sorted(
        RESULTS_DIR.glob("*.json"), key=lambda item: natural_key(item.stem)
    ):
        payload = json.loads(path.read_text())
        metadata = payload["hgp_benchmark"]
        version = metadata["version"]
        if metadata.get("name"):
            version = f"{version}-{metadata['name']}"

        for row in aggregate_scenarios(payload):
            row["version"] = version
            machines[metadata["machine"]].append(row)
    return machines


def print_report(machines: dict[str, list[dict]]) -> None:
    for machine in sorted(machines, key=natural_key):
        print(f"Machine: {machine}")
        rows = sorted(
            machines[machine],
            key=lambda row: (
                natural_key(row["scenario"]),
                natural_key(row["version"]),
            ),
        )
        table_rows = [
            {
                "Scenario": row["scenario"],
                "Version": row["version"],
                "Time": format_duration(row["duration"]),
                "Result": format_result(row["score_mean"], row["score_std"]),
            }
            for row in rows
        ]
        print(pd.DataFrame(table_rows).to_string(index=False))
        print()


def main() -> None:
    machines = load_results()
    if not machines:
        raise SystemExit(f"No benchmark results found in {RESULTS_DIR}")
    print_report(machines)


if __name__ == "__main__":
    main()
