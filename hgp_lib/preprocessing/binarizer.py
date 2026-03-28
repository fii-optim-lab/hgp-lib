from typing import Optional, Set

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.tree import DecisionTreeClassifier

from hgp_lib.utils.validation import check_isinstance


class StandardBinarizer:
    """
    Converts a mixed-type DataFrame into a purely boolean DataFrame.

    Boolean columns are passed through unchanged. Categorical columns are one-hot encoded
    into one boolean column per unique value. Numeric columns are discretised into bins and
    then one-hot encoded, using either quantile-based or decision-tree-based binning depending
    on whether target labels are supplied.

    After ``fit_transform`` the binarizer stores the learned bin edges and categorical
    mappings so that ``transform`` can apply the same encoding to new data.

    Args:
        num_bins (int):
            Default number of bins for numeric columns. Must be >= 2. Default: `5`.
        column_strategy (dict[str, int] | None):
            Per-column override for the number of bins. Keys are column names, values are
            the desired bin count (each >= 2). Default: `None`.
        precision (int):
            Number of decimal places used when formatting numeric bin boundary names.
            Must be >= 0. Default: `3`.

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
        column_strategy: Optional[dict[str, int]] = None,
        precision: int = 3,
    ):
        self._validate_params(num_bins, column_strategy, precision)
        self.num_bins = num_bins
        self.column_strategy = column_strategy or {}
        self.precision = precision
        self.column_precision: dict[str, int] = {}

        self._categorical_values: dict = {}
        self._numerical_bins: dict = {}
        self._columns = tuple()
        self._is_fitted = False

    def _validate_params(
        self, num_bins: int, column_strategy: Optional[dict[str, int]], precision: int
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

    def _get_tree_based_bins(
        self, X: np.ndarray, y: np.ndarray, n_bins: int
    ) -> np.ndarray:
        """
        Compute bin edges for a single numeric feature using a decision-tree classifier.

        The tree is trained to predict ``y`` from ``X`` and its internal split thresholds
        become the bin boundaries. If ``X`` has one or fewer unique values, a single
        ``[-inf, inf]`` bin is returned.

        Args:
            X (np.ndarray):
                1-D array of feature values.
            y (np.ndarray):
                1-D array of target labels, same length as ``X``.
            n_bins (int):
                Maximum number of bins (passed as ``max_leaf_nodes`` to the tree).

        Returns:
            np.ndarray: Sorted bin edges starting with ``-inf`` and ending with ``inf``.

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.preprocessing import StandardBinarizer
            >>> b = StandardBinarizer()
            >>> bins = b._get_tree_based_bins(
            ...     np.array([1.0, 2.0, 3.0, 4.0]),
            ...     np.array([0, 0, 1, 1]),
            ...     n_bins=2,
            ... )
            >>> bins.tolist()
            [-inf, 2.5, inf]
        """
        if len(np.unique(X)) <= 1:
            return np.array([-np.inf, np.inf])

        tree = DecisionTreeClassifier(max_leaf_nodes=n_bins)
        tree.fit(X.reshape(-1, 1), y)

        thresholds = tree.tree_.threshold[tree.tree_.feature == 0]
        thresholds = np.sort(thresholds[thresholds != -2])

        return np.concatenate([[-np.inf], thresholds, [np.inf]])

    def _get_quantile_based_bins(self, X: np.ndarray, n_bins: int) -> np.ndarray:
        """
        Compute bin edges for a single numeric feature using quantile percentiles.

        Edges are placed at evenly spaced quantiles. Duplicate edges are removed so the
        actual number of bins may be fewer than ``n_bins`` when many values are identical.
        If ``X`` has one or fewer unique values, a single ``[-inf, inf]`` bin is returned.

        Args:
            X (np.ndarray):
                1-D array of feature values.
            n_bins (int):
                Desired number of bins.

        Returns:
            np.ndarray: Sorted unique bin edges starting with ``-inf`` and ending with ``inf``.

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.preprocessing import StandardBinarizer
            >>> b = StandardBinarizer()
            >>> bins = b._get_quantile_based_bins(np.array([1.0, 2.0, 3.0, 4.0]), n_bins=2)
            >>> bins.tolist()
            [-inf, 2.5, inf]
        """
        if len(np.unique(X)) <= 1:
            return np.array([-np.inf, np.inf])

        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(X, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        return np.unique(bins)

    def _ensure_unique_column_names(
        self, column_names: Set[str], new_column_name: str
    ) -> str:
        """
        Register ``new_column_name`` in ``column_names``, appending a numeric suffix if the
        name already exists to avoid collisions.

        The set is mutated in place: the chosen (possibly suffixed) name is added before
        returning.

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
        if new_column_name not in column_names:
            column_names.add(new_column_name)
            return new_column_name
        for i in range(1_000):
            version_i = f"{new_column_name}_{i}"
            if version_i not in column_names:
                column_names.add(version_i)
                return version_i
        # If we didn't find a unique column name by trying 1000 indices,
        # We will use a random string extension until we find a string we didn't visit before
        new_column_name = new_column_name + "_rand"
        while True:
            random_i = np.random.randint(len(column_names))
            new_column_name = f"{new_column_name}_{random_i}"
            if new_column_name not in column_names:
                column_names.add(new_column_name)
                return new_column_name

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Learn the binarisation mapping from ``X`` (and optionally ``y``) and return the
        transformed boolean DataFrame.

        When ``y`` is provided, numeric columns are binned using a decision-tree strategy
        that maximises class separation. Otherwise, quantile-based binning is used.

        Args:
            X (pd.DataFrame):
                Input DataFrame whose columns are boolean, categorical, or numeric.
            y (np.ndarray | None):
                Optional target labels used for supervised (tree-based) binning of numeric
                columns. Default: `None`.

        Returns:
            pd.DataFrame: A DataFrame with only boolean columns.

        Raises:
            TypeError: If ``X`` is not a DataFrame.
            ValueError: If a column has an unsupported dtype.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from hgp_lib.preprocessing import StandardBinarizer
            >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
            >>> binarizer = StandardBinarizer(num_bins=2)
            >>> result = binarizer.fit_transform(df)
            >>> result.shape
            (4, 2)
            >>> all(result.dtypes == bool)
            True
        """
        check_isinstance(X, pd.DataFrame)
        columns = {}
        column_names = set()

        for column in X.columns:
            if is_bool_dtype(X[column]):
                new_column_name = self._ensure_unique_column_names(column_names, column)
                columns[new_column_name] = X[column]

            elif isinstance(X[column].dtype, pd.CategoricalDtype):
                unique_values = X[column].unique()
                self._categorical_values[column] = unique_values
                for value in unique_values:
                    new_column_name = self._ensure_unique_column_names(
                        column_names, f"{column}={value}"
                    )
                    columns[new_column_name] = X[column] == value

            elif is_numeric_dtype(X[column]):
                n_bins = self.column_strategy.get(column, self.num_bins)

                bins = (
                    self._get_tree_based_bins(X[column].values, y, n_bins)
                    if y is not None
                    else self._get_quantile_based_bins(X[column].values, n_bins)
                )

                self._numerical_bins[column] = bins

                binned_values = pd.cut(
                    X[column], bins=bins, labels=False, include_lowest=True
                )

                for bin_idx in range(len(bins) - 1):
                    new_column_name = self._format_numeric_bin_name(
                        column, bins[bin_idx], bins[bin_idx + 1]
                    )
                    new_column_name = self._ensure_unique_column_names(
                        column_names, new_column_name
                    )
                    columns[new_column_name] = binned_values == bin_idx

            else:
                raise ValueError(
                    f"Unsupported column type for column {column} of type {X[column].dtype}"
                )

        self._original_columns = X.columns
        self._columns = tuple(columns.keys())
        self._is_fitted = True
        return pd.DataFrame(columns, index=X.index)

    def _format_numeric_bin_name(self, column: str, left: float, right: float) -> str:
        """
        Build a human-readable label for a numeric bin.

        The format depends on whether the left or right boundary is infinite:

        - Left is ``-inf``: ``"column < right"``
        - Right is ``inf``: ``"left <= column"``
        - Both finite: ``"left <= column < right"``

        Boundary values are formatted using the precision configured for the column
        (falling back to ``self.precision``).

        Args:
            column (str):
                Name of the original numeric column.
            left (float):
                Left (inclusive) boundary of the bin.
            right (float):
                Right (exclusive) boundary of the bin.

        Returns:
            str: Formatted bin label.

        Examples:
            >>> from hgp_lib.preprocessing import StandardBinarizer
            >>> import numpy as np
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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the previously learned binarisation to new data.

        The binarizer must have been fitted via ``fit_transform`` before calling this method.
        The input DataFrame must have the same columns (in the same order and with the same
        dtypes) as the one used during fitting.

        Args:
            X (pd.DataFrame):
                Input DataFrame with the same schema as the fitting data.

        Returns:
            pd.DataFrame: A boolean DataFrame with the same column layout as the fitted output.

        Raises:
            TypeError: If ``X`` is not a DataFrame.
            ValueError: If the binarizer has not been fitted yet, or if a column has an
                unsupported dtype.

        Examples:
            >>> import pandas as pd
            >>> from hgp_lib.preprocessing import StandardBinarizer
            >>> train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
            >>> binarizer = StandardBinarizer(num_bins=2)
            >>> _ = binarizer.fit_transform(train)
            >>> test = pd.DataFrame({"x": [1.5, 3.5]})
            >>> result = binarizer.transform(test)
            >>> result.shape
            (2, 2)
        """
        check_isinstance(X, pd.DataFrame)

        if not self._is_fitted:
            raise ValueError("Binarizer must be fitted before calling transform")
        if not self._original_columns.equals(X.columns):
            raise RuntimeError(
                f"Original columns do not match current columns. "
                f"Original columns: {self._original_columns}. Current columns: {X.columns}."
            )

        columns = {}
        column_index = 0

        for column in X.columns:
            if is_bool_dtype(X[column]):
                column_name = self._columns[column_index]
                column_index += 1
                columns[column_name] = X[column]

            elif isinstance(X[column].dtype, pd.CategoricalDtype):
                for value in self._categorical_values[column]:
                    column_name = self._columns[column_index]
                    column_index += 1
                    columns[column_name] = X[column] == value

            elif is_numeric_dtype(X[column]):
                bins = self._numerical_bins[column]
                binned_values = pd.cut(
                    X[column], bins=bins, labels=False, include_lowest=True
                )
                for bin_idx in range(len(bins) - 1):
                    column_name = self._columns[column_index]
                    column_index += 1
                    columns[column_name] = binned_values == bin_idx

            else:
                raise ValueError(
                    f"Unsupported column type for column {column} of type {X[column].dtype}"
                )

        return pd.DataFrame(columns, index=X.index)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
