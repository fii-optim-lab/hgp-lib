import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--rule_artifacts", action="store_true")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

    from benchmarks.rule_artifacts import create_rule_artifacts

    if args.rule_artifacts or args.all:
        create_rule_artifacts(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
