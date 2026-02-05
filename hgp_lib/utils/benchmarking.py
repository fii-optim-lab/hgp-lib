"""
Benchmarking utilities for Boolean GP.

This module provides functions for loading and preprocessing datasets
for use with the Boolean GP benchmarking.
"""

import gc
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from numpy import ndarray

from hgp_lib.preprocessing import StandardBinarizer

logger = logging.getLogger(__name__)


def load_data(data_path: str, num_bins: int = 5) -> Tuple[ndarray, ndarray]:
    """
    Load and preprocess data from HDF file for benchmarking.

    Loads the data, identifies the target column, and binarizes features
    using StandardBinarizer. Supports PaySim format (isFraud column) and
    generic format (target column).

    Args:
        data_path: Path to HDF file containing data.
        num_bins: Number of bins for binarization. Default: 5.

    Returns:
        Tuple of (data, labels) as numpy arrays, binarized and ready for GP.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If target column cannot be identified.

    Examples:
        >>> from hgp_lib.utils.benchmarking import load_data
        >>> data, labels = load_data("data/PaySim.hdf")  # doctest: +SKIP
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from {data_path}...")

    df: pd.DataFrame = pd.read_hdf(data_path)

    # Detect target column - PaySim uses isFraud, others use target
    if "isFraud" in df.columns:
        target_column = "isFraud"
    elif "target" in df.columns:
        target_column = "target"
    else:
        raise RuntimeError(f"Unknown target column. Available: {df.columns.tolist()}")

    labels = df[target_column].values.copy()
    data = df.drop([target_column], axis=1)

    del df

    logger.info(f"Loaded {len(data)} samples with {len(data.columns)} features")
    logger.info(f"Positive rate: {labels.mean():.4f} ({labels.sum()} positive cases)")

    logger.info("Binarizing data...")
    binarizer = StandardBinarizer(num_bins=num_bins)
    data_bin = binarizer.fit_transform(data, labels)

    logger.info(f"Binarized features: {data_bin.shape[1]} (from {data.shape[1]})")

    data_bin = data_bin.values

    del data
    gc.collect()

    return data_bin, labels
