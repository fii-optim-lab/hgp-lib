import argparse
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from time import sleep


def get_binary_classification_datasets():
    from pmlb.dataset_lists import df_summary

    return df_summary[df_summary["n_classes"] == 2.0]["dataset"].tolist()


def get_commands_for_datasets(dataset_names):
    commands = []
    for dataset_name in dataset_names:
        cmd = (
            f"{sys.executable} "
            "scripts/optuna_hypertuning.py "
            f"--data-path data/{dataset_name}.hdf "
            "--n-trials 25 "
            "--max-n-trials 25 "
            f"--study-name pmlb_{dataset_name} "
            "--artifact-dir ./artifacts"
        )
        # TODO: Implement skipping experiment if there already are n_trials experiments
        commands.append(cmd)
    return commands


def subprocess_runner(command: str):
    sleep(random.randint(1, 5))
    subprocess.run(command.split(" "), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_jobs", type=int, default=4)
    args = parser.parse_args()

    dataset_names = get_binary_classification_datasets()
    commands = get_commands_for_datasets(dataset_names)
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        executor.map(subprocess_runner, commands)


if __name__ == "__main__":
    main()
