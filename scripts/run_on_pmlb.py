import subprocess


def get_binary_classification_datasets():
    from pmlb.dataset_lists import df_summary

    return df_summary[df_summary["n_classes"] == 2.0]["dataset"].tolist()


def main():
    dataset_names = get_binary_classification_datasets()

    for dataset_name in dataset_names:
        print(f"Running on {dataset_name}")
        cmd = (
            "python "
            "scripts/optuna_hypertuning.py "
            f"--data-path data/{dataset_name}.hdf "
            "--n-trials 100 "
            f"--study-name pmlb_{dataset_name} "
            "--verbose --artifact-dir ./artifacts"
        )
        print(cmd)
        cmd = cmd.split(" ")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
