import logging
from typing import Tuple

import pandas as pd
from numpy import ndarray
from pathlib import Path


# TODO: Add tests for this method.
def load_data(data_path: str) -> Tuple[pd.DataFrame, ndarray]:
    """
    Load data and labels from a CSV/HDF file.
    The target column is assumed to be named "target".
    All columns before the target column are considered features.

    # TODO: Make documentation consistent with the rest of the library.

    Args:
        data_path: Path to CSV/HDF file containing data.

    Returns:
        Tuple of (data, labels) as pandas DataFrames and numpy arrays.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If target column cannot be identified.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logging.getLogger(__name__).info(f"Loading data from {data_path}...")
    # TODO: Create a unified logging system

    if path.suffix == ".hdf":
        df: pd.DataFrame = pd.read_hdf(data_path)
    elif path.suffix == ".csv":
        df: pd.DataFrame = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    if "target" in df.columns:
        target_column = "target"
    else:
        raise RuntimeError(f"Unknown target column. Available: {df.columns.tolist()}")

    labels = df[target_column].to_numpy(dtype=bool, copy=True)
    data = df.drop([target_column], axis=1)

    del df
    return data, labels
