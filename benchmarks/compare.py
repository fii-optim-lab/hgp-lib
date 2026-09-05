import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark-results"


def natural_key(value: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def result_key(path: Path) -> tuple:
    return path.stem.lower() == "current", natural_key(path.stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        action="store_true",
        help="Show the score for every benchmark case.",
    )
    return parser.parse_args()


def load_result(path: Path) -> dict:
    payload = json.loads(path.read_text())
    scenarios = {}

    for benchmark in payload["benchmarks"]:
        info = benchmark.get("extra_info", {})
        scenario_id = info.get("scenario_id")
        case_id = info.get("case_id")
        if scenario_id is None or case_id is None:
            continue

        scenarios.setdefault(scenario_id, {})[case_id] = {
            "duration": float(benchmark["stats"]["median"]),
            "score": info.get("test_score"),
        }

    return {"label": path.stem, "scenarios": scenarios}


def aggregate(
    cases: dict, case_ids: list[str]
) -> tuple[float, float | None, float | None]:
    duration = sum(cases[case_id]["duration"] for case_id in case_ids)
    scores = [
        float(cases[case_id]["score"])
        for case_id in case_ids
        if cases[case_id]["score"] is not None
    ]
    if not scores:
        return duration, None, None
    return duration, statistics.mean(scores), statistics.pstdev(scores)


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.2f} s"


def format_change(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}{suffix}"


def format_score(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "-"
    return f"{mean:.4f} ± {std:.4f}"


def print_scores(cases: dict, case_ids: list[str]) -> None:
    scores = [
        f"{case_id}={float(cases[case_id]['score']):.4f}"
        for case_id in case_ids
        if cases[case_id]["score"] is not None
    ]
    if scores:
        print(f"  Scores: {', '.join(scores)}")


def print_comparison(results: list[dict], show_scores: bool = False) -> None:
    scenario_ids = sorted(
        {scenario_id for result in results for scenario_id in result["scenarios"]},
        key=natural_key,
    )

    for scenario_id in scenario_ids:
        baseline = next(
            result for result in results if scenario_id in result["scenarios"]
        )
        baseline_cases = baseline["scenarios"][scenario_id]
        baseline_case_ids = sorted(baseline_cases, key=natural_key)
        baseline_duration, baseline_score, _ = aggregate(
            baseline_cases, baseline_case_ids
        )

        print(f"Scenario: {scenario_id}")
        print(
            f"{'Result':<28} {'Cases':>7} {'Total time':>14} "
            f"{'Time change':>13} {'Score (mean ± std)':>22} {'Mean change':>13}"
        )

        for result in results:
            cases = result["scenarios"].get(scenario_id)
            if cases is None:
                print(f"{result['label']:<28} {'-':>7} {'-':>14}")
                continue

            case_ids = sorted(cases, key=natural_key)
            if case_ids != baseline_case_ids:
                print(
                    f"{result['label']:<28} {len(case_ids):>7} {'different cases':>14}"
                )
                continue

            duration, score, score_std = aggregate(cases, case_ids)
            time_change = 100 * (duration / baseline_duration - 1)
            score_change = (
                score - baseline_score
                if score is not None and baseline_score is not None
                else None
            )

            print(
                f"{result['label']:<28} {len(case_ids):>7} "
                f"{format_duration(duration):>14} "
                f"{format_change(time_change, '%'):>13} "
                f"{format_score(score, score_std):>22} "
                f"{format_change(score_change):>13}"
            )
            if show_scores:
                print_scores(cases, case_ids)
        print()


def main() -> None:
    args = parse_args()
    paths = sorted(RESULTS_DIR.glob("*.json"), key=result_key)
    if not paths:
        raise SystemExit(f"No benchmark results found in {RESULTS_DIR}")

    print_comparison(
        [load_result(path) for path in paths],
        show_scores=args.scores,
    )


if __name__ == "__main__":
    main()
