from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import StratifiedKFold


RANDOM_STATE = 42
N_SPLITS = 5

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


OPENML_DATASETS = {
    # "banknote_authentication": 1462,
    # "diabetes": 37,
    # "spambase": 44,
    # "ionosphere": 59,
}

DATASET_NAMES = [
    "breast_cancer",
    *OPENML_DATASETS,
]


def _encode_binary_target(y) -> np.ndarray:
    y = np.asarray(y)
    classes = np.unique(y)

    if len(classes) != 2:
        raise ValueError(
            f"Expected a binary target, found classes {classes!r}."
        )

    return y == classes[1]


def _download_dataset(
    name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if name == "breast_cancer":
        X, y = load_breast_cancer(
            return_X_y=True,
            as_frame=True,
        )
    else:
        X, y = fetch_openml(
            data_id=OPENML_DATASETS[name],
            return_X_y=True,
            as_frame=True,
        )

    X = X.astype(np.float64)
    y = _encode_binary_target(y)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            f"Dataset {name!r} contains non-finite values."
        )

    return X, y


def load_dataset(
    name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a dataset, downloading and caching it when necessary."""
    if name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset: {name!r}")

    path = DATA_DIR / f"{name}.npz"

    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        X, y = _download_dataset(name)

        np.savez_compressed(
            path,
            X=X.to_numpy(),
            columns=X.columns.to_numpy(dtype=str),
            y=y,
        )

    with np.load(path) as dataset:
        X = pd.DataFrame(
            dataset["X"],
            columns=dataset["columns"].tolist(),
        )
        y = dataset["y"]

    return X, y


def load_fold(
    name: str,
    fold: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """Load one deterministic stratified fold."""
    if not 0 <= fold < N_SPLITS:
        raise ValueError(
            f"Fold must be between 0 and {N_SPLITS - 1}."
        )

    X, y = load_dataset(name)

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        split
        for index, split in enumerate(splitter.split(X, y))
        if index == fold
    )

    return (
        X.iloc[train_indices].reset_index(drop=True),
        X.iloc[test_indices].reset_index(drop=True),
        y[train_indices],
        y[test_indices],
    )


def download_all_datasets() -> None:
    for name in DATASET_NAMES:
        load_dataset(name)


if __name__ == "__main__":
    download_all_datasets()

    for name in DATASET_NAMES:
        X, y = load_dataset(name)

        print(
            f"{name}: "
            f"samples={X.shape[0]}, "
            f"features={X.shape[1]}, "
            f"positive={y.sum()}, "
            f"negative={(~y).sum()}"
        )
        print(f"  columns={X.columns.tolist()}")