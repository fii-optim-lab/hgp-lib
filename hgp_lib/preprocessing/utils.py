import logging
from pathlib import Path

import pandas as pd
from numpy import ndarray
from pandas.api.types import is_object_dtype, is_string_dtype


def is_categorical_like(column: pd.Series) -> bool:
    """
    Return whether a column should be treated as categorical.

    A column is categorical-like when it uses the pandas ``category`` dtype, or holds
    strings or Python objects. Numeric and boolean columns are not categorical-like.

    Args:
        column (pd.Series):
            The column to inspect.

    Returns:
        bool: ``True`` if the column is categorical, string, or object dtype.

    Examples:
        >>> import pandas as pd
        >>> from hgp_lib.preprocessing.utils import is_categorical_like
        >>> is_categorical_like(pd.Series(pd.Categorical(["a", "b"])))
        True
        >>> is_categorical_like(pd.Series(["a", "b"], dtype="string"))
        True
        >>> is_categorical_like(pd.Series([1.0, 2.0]))
        False
        >>> is_categorical_like(pd.Series([True, False]))
        False
    """
    return (
        isinstance(column.dtype, pd.CategoricalDtype)
        or is_string_dtype(column)
        or is_object_dtype(column)
    )


def load_data(data_path: str) -> tuple[pd.DataFrame, ndarray]:
    """
    Load features and labels from a CSV or HDF file.

    The file must contain a column named ``"target"`` which is used as the label
    array. All other columns are returned as the feature DataFrame. Labels are
    cast to ``bool``.

    Args:
        data_path (str):
            Path to a ``.csv`` or ``.hdf`` file.

    Returns:
        tuple[pd.DataFrame, ndarray]: ``(data, labels)`` where ``data`` is the
        feature DataFrame (without the target column) and ``labels`` is a 1-D
        boolean numpy array.

    Raises:
        FileNotFoundError: If ``data_path`` does not exist.
        ValueError: If the file extension is not ``.csv`` or ``.hdf``.
        RuntimeError: If no ``"target"`` column is found.

    Examples:
        >>> import tempfile, os
        >>> import pandas as pd
        >>> from hgp_lib.preprocessing.utils import load_data
        >>> df = pd.DataFrame({"x": [1, 2, 3], "target": [1, 0, 1]})
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = os.path.join(d, "tmp.csv")
        ...     df.to_csv(path, index=False)
        ...     data, labels = load_data(path)
        >>> list(data.columns)
        ['x']
        >>> labels.tolist()
        [True, False, True]
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
