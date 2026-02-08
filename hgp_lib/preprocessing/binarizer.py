from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.tree import DecisionTreeClassifier

from hgp_lib.utils.validation import check_isinstance


class StandardBinarizer:
    """
    A binarizer that transforms mixed-type data into binary format.

    This binarizer:
    - Preserves boolean columns
    - Converts categorical columns to one-hot encoding
    - Bins numerical columns using:
        * Quantile-based approach for unlabeled data
        * Decision tree-based approach for labeled data

    Attributes:
        num_bins (int): Number of bins for numerical features. Default: `5`.
        column_strategy (dict): Custom binning strategy for specific columns.
        categorical_values_ (dict): Stores unique values for each categorical column.
        numerical_bins_ (dict): Stores bin edges for each numerical column.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data = pd.DataFrame({
        ...     'bool_col': [True, False, True, False],
        ...     'cat_col': pd.Categorical(['A', 'B', 'A', 'C']),
        ...     'num_col': [1.0, 2.0, 3.0, 4.0]
        ... })
        >>> binarizer = StandardBinarizer(num_bins=2)
        >>> result = binarizer.fit_transform(data)
        >>> result
           bool_col  cat_col=A  cat_col=B  cat_col=C  num_col < 2.500  2.500 <= num_col
        0      True       True      False      False             True             False
        1     False      False       True      False             True             False
        2      True       True      False      False            False              True
        3     False      False      False       True            False              True
    """

    def __init__(
        self,
        num_bins: int = 5,
        column_strategy: Optional[dict[str, int]] = None,
        precision: int = 3,
    ):
        """
        Initialize the StandardBinarizer.

        Args:
            num_bins (int): Number of bins for numerical features. Default: `5`.
            column_strategy (dict[str, int] | None): Custom binning strategy for specific columns.
                Format: {column_name: num_bins}. Default: `None`.
            precision (int): Number of decimals to be included in the column name for numerical columns.
                Default: `3`.

        Raises:
            ValueError: If num_bins is less than 2 or if column_strategy is invalid.

        Examples:
            >>> binarizer = StandardBinarizer(num_bins=3, column_strategy={'num_col': 4})
            >>> binarizer.num_bins
            3
            >>> binarizer.column_strategy
            {'num_col': 4}
        """
        self._validate_params(num_bins, column_strategy, precision)
        self.num_bins = num_bins
        self.column_strategy = column_strategy or {}
        self.precision = precision
        # TODO: Update documentation
        # TODO: Add precision strategy
        self.column_precision = {}
        self.categorical_values_ = {}
        self.numerical_bins_ = {}
        self._is_fitted = False

    def _validate_params(
        self, num_bins: int, column_strategy: Optional[dict[str, int]], precision: int
    ) -> None:
        """Validate initialization parameters."""
        check_isinstance(num_bins, int)
        if num_bins < 2:
            raise ValueError("num_bins must be an integer >= 2")

        if column_strategy is not None:
            check_isinstance(column_strategy, dict)
            for col, bins in column_strategy.items():
                check_isinstance(bins, int)
                if bins < 2:
                    raise ValueError(
                        f"Number of bins for column {col} must be an integer >= 2"
                    )

        # TODO: Add tests
        check_isinstance(precision, int)
        if precision < 0:
            raise ValueError("precision must be an integer >= 0")

    def _get_tree_based_bins(
        self, X: np.ndarray, y: np.ndarray, n_bins: int
    ) -> np.ndarray:
        """
        Get bin edges using decision tree splits.

        Args:
            X (np.ndarray): Input feature to be binned.
            y (np.ndarray): Target values for supervised binning.
            n_bins (int): Number of bins to create.

        Returns:
            numpy.ndarray: Array of bin edges including -inf and inf

        Examples:
            >>> X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
            >>> y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            >>> binarizer = StandardBinarizer(num_bins=2)
            >>> binarizer._get_tree_based_bins(X, y, 2)
            array([-inf,  4.5,  inf])
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
        Get bin edges using quantile-based approach.

        Args:
            X (np.ndarray): Input feature to be binned.
            n_bins (int): Number of bins to create.

        Returns:
            numpy.ndarray: Array of bin edges including -inf and inf

        Examples:
            >>> X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
            >>> binarizer = StandardBinarizer(num_bins=4)
            >>> binarizer._get_quantile_based_bins(X, 4)
            array([-inf, 2.75, 4.5 , 6.25,  inf])
        """
        if len(np.unique(X)) <= 1:
            return np.array([-np.inf, np.inf])

        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(X, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        return np.unique(bins)

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Fit the binarizer and transform the input data.

        Args:
            X (pd.DataFrame): Input data to be transformed.
            y (np.ndarray | None): Target values for supervised binning of numerical features. Default: `None`.

        Returns:
            pandas.DataFrame: Transformed binary data

        Raises:
            ValueError: If X is not a pandas DataFrame

        Examples:
            >>> data = pd.DataFrame({
            ...     'bool_col': [True, False, True, False],
            ...     'cat_col': pd.Categorical(['A', 'B', 'A', 'C']),
            ...     'num_col': [1.0, 2.0, 3.0, 4.0]
            ... })
            >>> binarizer = StandardBinarizer(num_bins=2)
            >>> result = binarizer.fit_transform(data)
            >>> result.columns.tolist()
            ['bool_col', 'cat_col=A', 'cat_col=B', 'cat_col=C', 'num_col < 2.500', '2.500 <= num_col']
            >>> result.dtypes.unique()
            array([dtype('bool')], dtype=object)
        """
        check_isinstance(X, pd.DataFrame)
        result = pd.DataFrame(index=X.index)

        for column in X.columns:
            if is_bool_dtype(X[column]):
                result[column] = X[column]

            elif isinstance(X[column].dtype, pd.CategoricalDtype):
                unique_values = X[column].unique()
                self.categorical_values_[column] = unique_values
                for value in unique_values:
                    result[f"{column}={value}"] = X[column] == value

            elif is_numeric_dtype(X[column]):
                n_bins = self.column_strategy.get(column, self.num_bins)

                bins = (
                    self._get_tree_based_bins(X[column].values, y, n_bins)
                    if y is not None
                    else self._get_quantile_based_bins(X[column].values, n_bins)
                )

                self.numerical_bins_[column] = bins

                binned_values = pd.cut(
                    X[column], bins=bins, labels=False, include_lowest=True
                )
                for bin_idx in range(len(bins) - 1):
                    result[
                        self._format_numeric_bin_name(
                            column, bins[bin_idx], bins[bin_idx + 1]
                        )
                    ] = binned_values == bin_idx

            else:
                raise ValueError(
                    f"Unsupported column type for column {column} of type {X[column].dtype}"
                )

        self._is_fitted = True
        return result

    def _format_numeric_bin_name(self, column: str, left: float, right: float) -> str:
        precision = self.column_precision.get(column, self.precision)
        if np.isneginf(left):
            return f"{column} < {right:.{precision}f}"
        if np.isposinf(right):
            return f"{left:.{precision}f} <= {column}"
        return f"{left:.{precision}f} <= {column} < {right:.{precision}f}"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted binarizer.

        Args:
            X (pd.DataFrame): Input data to transform.

        Returns:
            pandas.DataFrame: Transformed binary data

        Raises:
            ValueError: If X is not a pandas DataFrame or if binarizer is not fitted

        Examples:
            >>> train_data = pd.DataFrame({
            ...     'bool_col': [True, False, True, False],
            ...     'cat_col': pd.Categorical(['A', 'B', 'A', 'C']),
            ...     'num_col': [1.0, 2.0, 3.0, 4.0]
            ... })
            >>> binarizer = StandardBinarizer(num_bins=2)
            >>> _ = binarizer.fit_transform(train_data)
            >>> new_data = pd.DataFrame({
            ...     'bool_col': [True, False],
            ...     'cat_col': pd.Categorical(['B', 'C']),
            ...     'num_col': [1.5, 3.5]
            ... })
            >>> result = binarizer.transform(new_data)
            >>> result
               bool_col  cat_col=A  cat_col=B  cat_col=C  num_col < 2.500  2.500 <= num_col
            0      True      False       True      False             True             False
            1     False      False      False       True            False              True
        """
        check_isinstance(X, pd.DataFrame)

        if not self._is_fitted:
            raise ValueError("Binarizer must be fitted before calling transform")

        result = pd.DataFrame(index=X.index)

        for column in X.columns:
            if is_bool_dtype(X[column]):
                result[column] = X[column]

            elif isinstance(X[column].dtype, pd.CategoricalDtype):
                for value in self.categorical_values_[column]:
                    result[f"{column}={value}"] = X[column] == value

            elif is_numeric_dtype(X[column]):
                bins = self.numerical_bins_[column]
                binned_values = pd.cut(
                    X[column], bins=bins, labels=False, include_lowest=True
                )
                for bin_idx in range(len(bins) - 1):
                    result[
                        self._format_numeric_bin_name(
                            column, bins[bin_idx], bins[bin_idx + 1]
                        )
                    ] = binned_values == bin_idx

            else:
                raise ValueError(
                    f"Unsupported column type for column {column} of type {X[column].dtype}"
                )

        return result


if __name__ == "__main__":
    import doctest

    doctest.testmod()
