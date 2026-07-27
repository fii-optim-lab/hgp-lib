
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from tqdm import tqdm

from hgp_lib.utils.validation import check_isinstance
from hgp_lib.utils.warnings import warn_once

from .base import Binarizer
from .binning import BinningStrategy, QuantileBinning, SupervisedTreeBinning
from .utils import is_categorical_like
from .warnings import (
    EmptyBinarizationWarning,
    HighCardinalityWarning,
    StringColumnWarning,
    UnseenNaNWarning,
)


class StandardBinarizer(Binarizer):
    """
    Converts a mixed-type DataFrame into a purely boolean DataFrame.

    Boolean columns are passed through unchanged. Categorical columns are one-hot
    encoded into one boolean column per unique value. Numeric columns are discretised
    into bins and then one-hot encoded, using a :class:`BinningStrategy`.

    Column handling:

    - Boolean columns are kept as is.
    - Categorical, string, and object columns are one-hot encoded. String and object
      columns trigger a :class:`StringColumnWarning`, since setting a ``category``
      dtype is clearer. A column whose values are all distinct is dropped with a
      :class:`HighCardinalityWarning`, because one-hot encoding it carries no
      generalization.
    - Numeric columns are split into bins by a :class:`BinningStrategy`. When ``y`` is
      provided and no strategy is set, :class:`SupervisedTreeBinning` is used, otherwise
      :class:`QuantileBinning`.
    - A column that contains missing values also gets a boolean ``<col>_is_NA``
      indicator column.

    To change how numeric bins are chosen, pass a ``numeric_binning`` strategy or
    subclass and override ``_fit_numeric`` / ``_transform_numeric``. Categorical and
    boolean handling can be changed the same way through their ``_fit_*`` / ``_transform_*``
    hooks.

    Args:
        num_bins (int):
            Default number of bins for numeric columns. Must be >= 2. Default: `5`.
        column_strategy (dict[str, int] | None):
            Per-column override for the number of bins. Keys are column names, values are
            the desired bin count (each >= 2). Default: `None`.
        precision (int):
            Number of decimal places used when formatting numeric bin boundary names.
            Must be >= 0. Default: `3`.
        numeric_binning (BinningStrategy | None):
            Strategy for computing numeric bin edges. When `None`, the binarizer uses
            :class:`SupervisedTreeBinning` if labels are provided to ``fit_transform``
            and :class:`QuantileBinning` otherwise. Default: `None`.
        progress_bar (bool):
            Whether to show progress bar. Default: `True`.
        leave_progress_bar (bool):
            Whether to leave progress bar. Default: `False`.


    Examples:
        >>> import pandas as pd
        >>> from hgp_lib.preprocessing import StandardBinarizer
        >>> df = pd.DataFrame({"flag": [True, False, True], "val": [1.0, 2.0, 3.0]})
        >>> binarizer = StandardBinarizer(num_bins=2)
        >>> result = binarizer.fit_transform(df)
        >>> "flag" in result.columns
        True
        >>> result["flag"].tolist()
        [True, False, True]
        >>> result
            flag  val < 2.000  2.000 <= val
        0   True         True         False
        1  False         True         False
        2   True        False          True
    """

    def __init__(
        self,
        num_bins: int = 5,
        column_strategy: dict[str, int] | None = None,
        precision: int = 3,
        numeric_binning: BinningStrategy | None = None,
        progress_bar: bool = True,
        leave_progress_bar: bool = False,
    ):
        self._validate_params(num_bins, column_strategy, precision, numeric_binning)
        self.num_bins = num_bins
        self.column_strategy = column_strategy or {}
        self.precision = precision
        self.numeric_binning = numeric_binning
        self.column_precision: dict[str, int] = {}
        self.progress_bar = progress_bar
        self.leave_progress_bar = leave_progress_bar

        self._categorical_values: dict = {}
        self._numerical_bins: dict = {}
        self._original_column_dtypes: dict = {}
        self._output_names: dict = {}
        self._na_columns: set[str] = set()
        self._skipped_columns: set[str] = set()
        self._original_columns = None
        self._is_fitted = False
        self._feature_names: list[str] = []

    def _validate_params(
        self,
        num_bins: int,
        column_strategy: dict[str, int] | None,
        precision: int,
        numeric_binning: BinningStrategy | None,
    ) -> None:
        check_isinstance(num_bins, int)
        if num_bins < 2:
            raise ValueError(f"num_bins must be an integer >= 2, is {num_bins}")

        if column_strategy is not None:
            check_isinstance(column_strategy, dict)
            for col, bins in column_strategy.items():
                check_isinstance(bins, int)
                if bins < 2:
                    raise ValueError(
                        f"Number of bins for column {col} must be an integer >= 2, is {bins}"
                    )

        check_isinstance(precision, int)
        if precision < 0:
            raise ValueError(f"precision must be an integer >= 0, is {precision}")

        if numeric_binning is not None:
            check_isinstance(numeric_binning, BinningStrategy)

    def fit_transform(
        self, X: pd.DataFrame, y: np.ndarray | None = None
    ) -> pd.DataFrame:
        """
        Learn the binarisation mapping from ``X`` (and optionally ``y``) and return the
        transformed boolean DataFrame.

        Args:
            X (pd.DataFrame):
                Input DataFrame whose columns are boolean, categorical, string, object,
                or numeric.
            y (np.ndarray | None):
                Optional target labels used for supervised binning of numeric columns.
                Default: `None`.

        Returns:
            pd.DataFrame: A DataFrame with only boolean columns.

        Raises:
            TypeError: If ``X`` is not a DataFrame.
            ValueError: If a column has an unsupported dtype.

        Examples:
            >>> import pandas as pd
            >>> from hgp_lib.preprocessing import StandardBinarizer
            >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
            >>> result = StandardBinarizer(num_bins=2).fit_transform(df)
            >>> result.shape
            (4, 2)
            >>> all(result.dtypes == bool)
            True
        """
        check_isinstance(X, pd.DataFrame)
        self._reset_state()

        outputs: dict = {}
        used_names: set[str] = set()

        for column in tqdm(
            X.columns,
            disable=not self.progress_bar,
            desc="Fitting binarizer",
            leave=self.leave_progress_bar,
        ):
            series = X[column]
            nan_mask = series.isna().to_numpy()

            pieces = []
            if nan_mask.any():
                self._na_columns.add(column)
                pieces.append((f"{column}_is_NA", nan_mask))

            pieces.extend(self._fit_column(column, series, y, nan_mask))

            names: list[str] = []
            for base_name, values in pieces:
                name = self._ensure_unique_column_names(used_names, base_name)
                outputs[name] = values
                names.append(name)
            self._output_names[column] = names

        if len(outputs) == 0:
            warn_once(EmptyBinarizationWarning())
            outputs["default"] = np.ones(len(X), dtype=bool)

        self._original_columns = X.columns
        self._is_fitted = True
        self._feature_names = [str(name) for name in outputs]
        return pd.DataFrame(outputs, index=X.index)

    def get_feature_names_out(self) -> list[str]:
        """
        Return the output column names in order (see :meth:`Binarizer.get_feature_names_out`).

        Returns:
            List[str]: The boolean output column names, index-aligned with the
                columns produced by ``fit_transform`` / ``transform``.

        Raises:
            ValueError: If the binarizer has not been fitted yet.

        Examples:
            >>> import pandas as pd
            >>> from hgp_lib.preprocessing import StandardBinarizer
            >>> b = StandardBinarizer(num_bins=2)
            >>> _ = b.fit_transform(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}))
            >>> b.get_feature_names_out()
            ['x < 2.500', '2.500 <= x']
        """
        if not self._is_fitted:
            raise ValueError(
                "Binarizer must be fitted before calling get_feature_names_out"
            )
        return list(self._feature_names)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the previously learned binarisation to new data.

        The input must have the same columns, in the same order and with the same
        dtypes, as the data used during fitting.

        Args:
            X (pd.DataFrame):
                Input DataFrame with the same schema as the fitting data.

        Returns:
            pd.DataFrame: A boolean DataFrame with the same column layout as the fitted output.

        Raises:
            TypeError: If ``X`` is not a DataFrame.
            ValueError: If the binarizer has not been fitted yet, or if a column dtype
                differs from the one seen during fitting.
            RuntimeError: If the columns differ from the fitting data.

        Examples:
            >>> import pandas as pd
            >>> from hgp_lib.preprocessing import StandardBinarizer
            >>> b = StandardBinarizer(num_bins=2)
            >>> _ = b.fit_transform(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}))
            >>> b.transform(pd.DataFrame({"x": [1.5, 3.5]})).shape
            (2, 2)
        """
        check_isinstance(X, pd.DataFrame)
        if not self._is_fitted:
            raise ValueError("Binarizer must be fitted before calling transform")
        if not self._original_columns.equals(X.columns):
            # TODO: We should add custom Errors in the library where it makes sense.;
            raise RuntimeError(
                f"Original columns do not match current columns. "
                f"Original columns: {self._original_columns}. Current columns: {X.columns}."
            )

        outputs: dict = {}
        for column in tqdm(
            X.columns,
            disable=not self.progress_bar,
            desc="Transforming binarizer",
            leave=self.leave_progress_bar,
        ):
            series = X[column]
            names = self._output_names[column]
            values_list: list[np.ndarray] = []

            nan_mask = series.isna().to_numpy()
            if column in self._na_columns:
                values_list.append(nan_mask)
            elif nan_mask.any():
                warn_once(UnseenNaNWarning(column))

            if column not in self._skipped_columns:
                values_list.extend(self._transform_column(column, series))

            if len(values_list) != len(names):
                raise RuntimeError(
                    f"Column '{column}' produced {len(values_list)} features at transform "
                    f"but {len(names)} were produced at fit."
                )
            for name, values in zip(names, values_list):
                outputs[name] = values

        if len(outputs) == 0:
            warn_once(EmptyBinarizationWarning())
            outputs["default"] = np.ones(len(X), dtype=bool)

        return pd.DataFrame(outputs, index=X.index)

    def _fit_column(
        self,
        column: str,
        series: pd.Series,
        y: np.ndarray | None,
        nan_mask: np.ndarray,
    ):
        """Dispatch a single column to the matching dtype hook and record its dtype."""
        # TODO: Instead of string values, we should have an enum. And an enum-like dispatch.
        if is_bool_dtype(series):
            self._original_column_dtypes[column] = "bool"
            return self._fit_boolean(column, series)
        if is_categorical_like(series):
            self._original_column_dtypes[column] = "category"
            return self._fit_categorical(column, series)
        if is_numeric_dtype(series):
            self._original_column_dtypes[column] = "numeric"
            return self._fit_numeric(column, series, y, nan_mask)
        raise ValueError(
            f"Unsupported column type for column {column} of type {series.dtype}"
        )

    def _fit_boolean(self, column: str, series: pd.Series):
        """Pass a boolean column through as a single feature."""
        return [(column, series.to_numpy(dtype=bool))]

    def _fit_categorical(self, column: str, series: pd.Series):
        """One-hot encode a categorical, string, or object column."""
        if not isinstance(series.dtype, pd.CategoricalDtype):
            warn_once(StringColumnWarning(column))

        not_na = series.dropna()
        unique_values = not_na.unique()
        if len(unique_values) == len(not_na):
            warn_once(HighCardinalityWarning(column))
            self._skipped_columns.add(column)
            return []

        self._categorical_values[column] = unique_values
        return [
            (f"{column}={value}", (series == value).to_numpy())
            for value in unique_values
        ]

    def _fit_numeric(
        self,
        column: str,
        series: pd.Series,
        y: np.ndarray | None,
        nan_mask: np.ndarray,
    ):
        """Bin a numeric column and one-hot encode the bins."""
        n_bins = self.column_strategy.get(column, self.num_bins)
        values = series.to_numpy()

        fit_values = values
        fit_y = y
        if nan_mask.any():
            keep = ~nan_mask
            fit_values = values[keep]
            if fit_y is not None:
                fit_y = fit_y[keep]

        strategy = self._resolve_numeric_binning(y)
        edges = strategy.compute_edges(fit_values, fit_y, n_bins)
        self._numerical_bins[column] = edges

        binned = pd.cut(values, bins=edges, labels=False, include_lowest=True)
        return [
            (
                self._format_numeric_bin_name(column, edges[i], edges[i + 1]),
                binned == i,
            )
            for i in range(len(edges) - 1)
        ]

    def _transform_column(self, column: str, series: pd.Series) -> list[np.ndarray]:
        """Apply the learned encoding for a single column, verifying its dtype."""
        expected = self._original_column_dtypes[column]
        actual = self._infer_kind(column, series)
        if actual != expected:
            raise ValueError(
                f"Original column {column} was {expected}. "
                f"Current column is {actual}. Current column must be {expected}."
            )
        if expected == "bool":
            return self._transform_boolean(series)
        if expected == "category":
            return self._transform_categorical(column, series)
        return self._transform_numeric(column, series)

    def _transform_boolean(self, series: pd.Series) -> list[np.ndarray]:
        return [series.to_numpy(dtype=bool)]

    def _transform_categorical(
        self, column: str, series: pd.Series
    ) -> list[np.ndarray]:
        return [
            (series == value).to_numpy() for value in self._categorical_values[column]
        ]

    def _transform_numeric(self, column: str, series: pd.Series) -> list[np.ndarray]:
        edges = self._numerical_bins[column]
        binned = pd.cut(
            series.to_numpy(), bins=edges, labels=False, include_lowest=True
        )
        return [binned == i for i in range(len(edges) - 1)]

    def _resolve_numeric_binning(self, y: np.ndarray | None) -> BinningStrategy:
        """Pick the numeric binning strategy: the configured one, or a default by ``y``."""
        if self.numeric_binning is not None:
            return self.numeric_binning
        return SupervisedTreeBinning() if y is not None else QuantileBinning()

    def _infer_kind(self, column: str, series: pd.Series) -> str:
        if is_bool_dtype(series):
            return "bool"
        if is_categorical_like(series):
            return "category"
        if is_numeric_dtype(series):
            return "numeric"
        raise ValueError(
            f"Unsupported column type for column {column} of type {series.dtype}"
        )

    def _reset_state(self) -> None:
        self._categorical_values = {}
        self._numerical_bins = {}
        self._original_column_dtypes = {}
        self._output_names = {}
        self._na_columns = set()
        self._skipped_columns = set()
        self._original_columns = None
        self._is_fitted = False

    def _ensure_unique_column_names(
        self, column_names: set[str], new_column_name: str
    ) -> str:
        """
        Register ``new_column_name`` in ``column_names``, appending a numeric suffix if needed.
        The set is mutated in place.

        Args:
            column_names (Set[str]):
                Mutable set of names already in use.
            new_column_name (str):
                Desired column name.

        Returns:
            str: The original name if it was unique, otherwise a suffixed variant.

        Examples:
            >>> from hgp_lib.preprocessing import StandardBinarizer
            >>> b = StandardBinarizer()
            >>> names = set(["col", "col_0"])
            >>> b._ensure_unique_column_names(names, "col")
            'col_1'
            >>> "col_1" in names
            True
        """
        unique_name = new_column_name
        counter = 0
        while unique_name in column_names:
            unique_name = f"{new_column_name}_{counter}"
            counter += 1
        column_names.add(unique_name)
        return unique_name

    def _format_numeric_bin_name(self, column: str, left: float, right: float) -> str:
        """
        Build a human-readable label for a numeric bin.

        The format depends on whether the left or right boundary is infinite:

        - Left is ``-inf``: ``"column < right"``
        - Right is ``inf``: ``"left <= column"``
        - Both finite: ``"left <= column < right"``

        Args:
            column (str):
                Name of the original numeric column.
            left (float):
                Left boundary of the bin.
            right (float):
                Right boundary of the bin.

        Returns:
            str: Formatted bin label.

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.preprocessing import StandardBinarizer
            >>> b = StandardBinarizer(precision=2)
            >>> b._format_numeric_bin_name("x", -np.inf, 3.0)
            'x < 3.00'
            >>> b._format_numeric_bin_name("x", 1.0, np.inf)
            '1.00 <= x'
            >>> b._format_numeric_bin_name("x", 1.0, 3.0)
            '1.00 <= x < 3.00'
        """
        precision = self.column_precision.get(column, self.precision)
        if np.isneginf(left):
            return f"{column} < {right:.{precision}f}"
        if np.isposinf(right):
            return f"{left:.{precision}f} <= {column}"
        return f"{left:.{precision}f} <= {column} < {right:.{precision}f}"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
