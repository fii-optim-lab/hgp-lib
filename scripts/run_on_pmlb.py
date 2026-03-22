import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def get_binary_classification_datasets():
    from pmlb.dataset_lists import df_summary

    return df_summary[df_summary["n_classes"] == 2.0]["dataset"].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_jobs", type=int, default=4)
    args = parser.parse_args()

    dataset_names = get_binary_classification_datasets()
    commands = []
    for dataset_name in dataset_names:
        cmd = (
            "python "
            "scripts/optuna_hypertuning.py "
            f"--data-path data/{dataset_name}.hdf "
            "--n-trials 100 "
            f"--study-name pmlb_{dataset_name} "
            "--artifact-dir ./artifacts"
        )
        commands.append(cmd.split(" "))

    subprocess_runner = partial(subprocess.run, check=True)
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        executor.map(subprocess_runner, commands)


if __name__ == "__main__":
    main()
