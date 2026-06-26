import argparse
import os

import optuna
import pandas as pd


def main(args):
    storage_path = args.storage_path
    prefix = args.prefix
    output_csv = args.output_csv
    if not os.path.exists(storage_path):
        raise FileNotFoundError(f"Storage path {storage_path} does not exist")
    storage = f"sqlite:///{storage_path}"
    summaries = optuna.get_all_study_summaries(storage=storage)

    total_studies = len(summaries)
    prefixed_studies = [s for s in summaries if s.study_name.startswith(prefix)]

    print(f"Total studies in DB: {total_studies}")
    print(f"Studies with prefix '{prefix}': {len(prefixed_studies)}")

    dataset_names = [s.study_name.removeprefix(prefix) for s in prefixed_studies]
    mean_test_scores = [
        s.best_trial.user_attrs.get("03_test_mean") for s in prefixed_studies
    ]
    std_test_scores = [
        s.best_trial.user_attrs.get("03_test_std") for s in prefixed_studies
    ]
    pd.DataFrame(
        [
            {"dataset": d, "mean_test_score": m, "std_test_score": s}
            for d, m, s in zip(dataset_names, mean_test_scores, std_test_scores)
        ]
    ).to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage_path",
        type=str,
        default="./optuna_study.db",
        help="Path to sqlite .db file",
    )
    parser.add_argument("--prefix", type=str, default="pmlb_", help="Study name prefix")
    parser.add_argument(
        "--output_csv", type=str, default="pmlb_gp.csv", help="Output csv file"
    )
    args = parser.parse_args()

    main(args)
