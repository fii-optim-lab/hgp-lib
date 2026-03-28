import doctest
import unittest

import numpy as np
import pandas as pd

import hgp_lib.preprocessing.binarizer
from hgp_lib.preprocessing import StandardBinarizer


class TestStandardBinarizer(unittest.TestCase):
    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #
    def test_num_bins_must_be_int(self):
        with self.assertRaises(TypeError):
            StandardBinarizer(num_bins=2.5)

    def test_num_bins_minimum(self):
        with self.assertRaises(ValueError):
            StandardBinarizer(num_bins=1)
        # boundary: 2 is valid
        StandardBinarizer(num_bins=2)

    def test_column_strategy_must_be_dict(self):
        with self.assertRaises(TypeError):
            StandardBinarizer(column_strategy=[3])

    def test_column_strategy_bins_must_be_int(self):
        with self.assertRaises(TypeError):
            StandardBinarizer(column_strategy={"a": 2.0})

    def test_column_strategy_bins_minimum(self):
        with self.assertRaises(ValueError):
            StandardBinarizer(column_strategy={"a": 1})

    def test_precision_must_be_int(self):
        with self.assertRaises(TypeError):
            StandardBinarizer(precision=1.0)

    def test_precision_minimum(self):
        with self.assertRaises(ValueError):
            StandardBinarizer(precision=-1)
        # boundary: 0 is valid
        StandardBinarizer(precision=0)

    def test_fit_transform_requires_dataframe(self):
        b = StandardBinarizer()
        with self.assertRaises(TypeError):
            b.fit_transform(np.array([[1, 2]]))

    def test_transform_requires_dataframe(self):
        b = StandardBinarizer()
        df = pd.DataFrame({"x": [1.0, 2.0]})
        b.fit_transform(df)
        with self.assertRaises(TypeError):
            b.transform(np.array([[1.0]]))

    def test_transform_before_fit_raises(self):
        b = StandardBinarizer()
        with self.assertRaises(ValueError):
            b.transform(pd.DataFrame({"x": [1.0]}))

    def test_unsupported_dtype_fit_transform(self):
        b = StandardBinarizer()
        df = pd.DataFrame({"s": pd.array(["a", "b"], dtype="string")})
        with self.assertRaises(ValueError, msg="Unsupported column type"):
            b.fit_transform(df)

    def test_unsupported_dtype_transform(self):
        b = StandardBinarizer()
        train = pd.DataFrame({"x": [True, False]})
        b.fit_transform(train)
        test = pd.DataFrame({"x": pd.array(["a", "b"], dtype="string")})
        with self.assertRaises(ValueError):
            b.transform(test)

    # ------------------------------------------------------------------ #
    #  Boolean columns
    # ------------------------------------------------------------------ #
    def test_bool_passthrough(self):
        df = pd.DataFrame({"flag": [True, False, True]})
        result = StandardBinarizer().fit_transform(df)
        self.assertEqual(list(result.columns), ["flag"])
        self.assertEqual(result["flag"].tolist(), [True, False, True])

    def test_bool_transform(self):
        train = pd.DataFrame({"flag": [True, False]})
        b = StandardBinarizer()
        b.fit_transform(train)
        test = pd.DataFrame({"flag": [False, True, True]})
        result = b.transform(test)
        self.assertEqual(result["flag"].tolist(), [False, True, True])

    # ------------------------------------------------------------------ #
    #  Categorical columns
    # ------------------------------------------------------------------ #
    def test_categorical_one_hot(self):
        df = pd.DataFrame({"color": pd.Categorical(["r", "g", "b", "r"])})
        result = StandardBinarizer().fit_transform(df)
        # one column per unique value
        self.assertEqual(result.shape, (4, 3))
        # first row is "r" → color=r should be True
        self.assertTrue(result.iloc[0]["color=r"])
        self.assertFalse(result.iloc[0]["color=g"])

    def test_categorical_transform(self):
        train = pd.DataFrame({"color": pd.Categorical(["r", "g", "b"])})
        b = StandardBinarizer()
        b.fit_transform(train)
        test = pd.DataFrame({"color": pd.Categorical(["g", "r"])})
        result = b.transform(test)
        self.assertEqual(result.shape[0], 2)
        self.assertTrue(result.iloc[0]["color=g"])

    # ------------------------------------------------------------------ #
    #  Numeric columns – quantile binning (no y)
    # ------------------------------------------------------------------ #
    def test_numeric_quantile_binning(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        b = StandardBinarizer(num_bins=2)
        result = b.fit_transform(df)
        # all output columns should be boolean
        for col in result.columns:
            self.assertTrue(result[col].dtype == bool)
        # each row should belong to exactly one bin
        self.assertTrue((result.sum(axis=1) == 1).all())

    def test_numeric_column_strategy_override(self):
        df = pd.DataFrame({"a": range(20), "b": range(20)})
        b = StandardBinarizer(num_bins=3, column_strategy={"a": 2})
        result = b.fit_transform(df)
        # "a" gets 2 bins, "b" gets 3 bins (default)
        # Column names contain the original column name in the bin label
        self.assertEqual(len(b._numerical_bins["a"]) - 1, 2)
        self.assertEqual(len(b._numerical_bins["b"]) - 1, 3)
        self.assertEqual(result.shape, (20, 5))

    def test_numeric_constant_column(self):
        """A column with a single unique value should produce one bin [-inf, inf]."""
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
        b = StandardBinarizer(num_bins=3)
        result = b.fit_transform(df)
        # only one bin possible
        self.assertEqual(result.shape[1], 1)
        self.assertTrue(result.iloc[0, 0])

    # ------------------------------------------------------------------ #
    #  Numeric columns – tree-based binning (with y)
    # ------------------------------------------------------------------ #
    def test_numeric_tree_binning(self):
        np.random.seed(42)
        df = pd.DataFrame({"x": np.arange(100, dtype=float)})
        y = (df["x"] > 50).astype(int).values
        b = StandardBinarizer(num_bins=3)
        result = b.fit_transform(df, y=y)
        for col in result.columns:
            self.assertTrue(result[col].dtype == bool)
        self.assertTrue((result.sum(axis=1) == 1).all())

    def test_tree_binning_constant_column(self):
        df = pd.DataFrame({"x": [7.0, 7.0, 7.0]})
        y = np.array([0, 1, 0])
        b = StandardBinarizer(num_bins=3)
        result = b.fit_transform(df, y=y)
        self.assertEqual(result.shape[1], 1)

    # ------------------------------------------------------------------ #
    #  _get_tree_based_bins
    # ------------------------------------------------------------------ #
    def test_tree_bins_constant_returns_single_bin(self):
        b = StandardBinarizer()
        bins = b._get_tree_based_bins(np.array([3.0, 3.0, 3.0]), np.array([0, 1, 0]), 3)
        np.testing.assert_array_equal(bins, [-np.inf, np.inf])

    def test_tree_bins_produces_sorted_edges(self):
        b = StandardBinarizer()
        X = np.arange(50, dtype=float)
        y = (X > 25).astype(int)
        bins = b._get_tree_based_bins(X, y, n_bins=3)
        self.assertTrue(np.all(np.diff(bins) > 0))

    def test_tree_bins_boundaries(self):
        b = StandardBinarizer()
        X = np.arange(20, dtype=float)
        y = (X >= 10).astype(int)
        bins = b._get_tree_based_bins(X, y, n_bins=2)
        self.assertEqual(bins[0], -np.inf)
        self.assertEqual(bins[-1], np.inf)
        self.assertGreaterEqual(len(bins), 3)  # at least one internal threshold

    def test_tree_bins_respects_max_bins(self):
        b = StandardBinarizer()
        X = np.arange(100, dtype=float)
        y = np.array([0, 1, 2, 3] * 25)
        bins = b._get_tree_based_bins(X, y, n_bins=4)
        # number of bins = len(edges) - 1, should be <= n_bins
        self.assertLessEqual(len(bins) - 1, 4)

    # ------------------------------------------------------------------ #
    #  _get_quantile_based_bins
    # ------------------------------------------------------------------ #
    def test_quantile_bins_constant_returns_single_bin(self):
        b = StandardBinarizer()
        bins = b._get_quantile_based_bins(np.array([5.0, 5.0, 5.0]), 3)
        np.testing.assert_array_equal(bins, [-np.inf, np.inf])

    def test_quantile_bins_produces_sorted_unique_edges(self):
        b = StandardBinarizer()
        bins = b._get_quantile_based_bins(np.arange(100, dtype=float), 4)
        self.assertTrue(np.all(np.diff(bins) > 0))

    def test_quantile_bins_boundaries(self):
        b = StandardBinarizer()
        bins = b._get_quantile_based_bins(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        self.assertEqual(bins[0], -np.inf)
        self.assertEqual(bins[-1], np.inf)

    def test_quantile_bins_deduplicates(self):
        """Many repeated values should collapse duplicate edges."""
        b = StandardBinarizer()
        X = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        bins = b._get_quantile_based_bins(X, n_bins=5)
        # np.unique removes duplicates, so we should have <= 5+1 edges
        self.assertLessEqual(len(bins), 6)
        # all edges unique
        self.assertEqual(len(bins), len(set(bins)))

    # ------------------------------------------------------------------ #
    #  _ensure_unique_column_names
    # ------------------------------------------------------------------ #
    def test_ensure_unique_new_name(self):
        b = StandardBinarizer()
        names: set[str] = set()
        result = b._ensure_unique_column_names(names, "col")
        self.assertEqual(result, "col")
        self.assertIn("col", names)

    def test_ensure_unique_collision_adds_suffix(self):
        b = StandardBinarizer()
        names: set[str] = {"col"}
        b._ensure_unique_column_names(names, "col")
        # A suffixed version should have been added
        self.assertTrue(any(n.startswith("col_") for n in names))

    def test_ensure_unique_multiple_collisions(self):
        b = StandardBinarizer()
        names: set[str] = set()
        b._ensure_unique_column_names(names, "x")
        b._ensure_unique_column_names(names, "x")
        b._ensure_unique_column_names(names, "x")
        # Should have 3 distinct entries
        self.assertEqual(len(names), 3)

    def test_ensure_unique_does_not_modify_unrelated(self):
        b = StandardBinarizer()
        names: set[str] = {"other"}
        b._ensure_unique_column_names(names, "col")
        self.assertIn("other", names)
        self.assertIn("col", names)

    # ------------------------------------------------------------------ #
    #  _format_numeric_bin_name
    # ------------------------------------------------------------------ #
    def test_format_left_inf(self):
        b = StandardBinarizer(precision=2)
        self.assertEqual(b._format_numeric_bin_name("v", -np.inf, 3.0), "v < 3.00")

    def test_format_right_inf(self):
        b = StandardBinarizer(precision=2)
        self.assertEqual(b._format_numeric_bin_name("v", 1.0, np.inf), "1.00 <= v")

    def test_format_both_finite(self):
        b = StandardBinarizer(precision=1)
        self.assertEqual(b._format_numeric_bin_name("v", 1.0, 3.0), "1.0 <= v < 3.0")

    def test_format_precision_zero(self):
        b = StandardBinarizer(precision=0)
        self.assertEqual(b._format_numeric_bin_name("x", 2.7, 5.3), "3 <= x < 5")

    def test_format_column_precision_overrides_default(self):
        b = StandardBinarizer(precision=5)
        b.column_precision["x"] = 1
        self.assertEqual(b._format_numeric_bin_name("x", 1.0, 2.0), "1.0 <= x < 2.0")
        # other columns still use default
        self.assertEqual(
            b._format_numeric_bin_name("y", 1.0, 2.0), "1.00000 <= y < 2.00000"
        )

    # ------------------------------------------------------------------ #
    #  Numeric bin name formatting (existing)
    # ------------------------------------------------------------------ #
    def test_bin_name_precision(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        b = StandardBinarizer(num_bins=2, precision=1)
        result = b.fit_transform(df)
        # leftmost bin should look like "x < ..."
        left_col = [c for c in result.columns if c.startswith("x <")]
        self.assertTrue(len(left_col) >= 1)

    def test_bin_name_column_precision_override(self):
        b = StandardBinarizer(precision=3)
        b.column_precision["x"] = 0
        name = b._format_numeric_bin_name("x", -np.inf, 2.5)
        self.assertEqual(name, "x < 2")

    def test_bin_name_left_inf(self):
        b = StandardBinarizer(precision=2)
        self.assertEqual(b._format_numeric_bin_name("v", -np.inf, 3.0), "v < 3.00")

    def test_bin_name_right_inf(self):
        b = StandardBinarizer(precision=2)
        self.assertEqual(b._format_numeric_bin_name("v", 1.0, np.inf), "1.00 <= v")

    def test_bin_name_finite(self):
        b = StandardBinarizer(precision=1)
        self.assertEqual(b._format_numeric_bin_name("v", 1.0, 3.0), "1.0 <= v < 3.0")

    # ------------------------------------------------------------------ #
    #  Mixed-type DataFrame
    # ------------------------------------------------------------------ #
    def test_mixed_types(self):
        df = pd.DataFrame(
            {
                "flag": [True, False, True, False],
                "color": pd.Categorical(["a", "b", "a", "b"]),
                "val": [10.0, 20.0, 30.0, 40.0],
            }
        )
        b = StandardBinarizer(num_bins=2)
        result = b.fit_transform(df)
        # bool(1) + categorical(2) + numeric(>=1)
        self.assertGreaterEqual(result.shape[1], 4)
        for col in result.columns:
            self.assertTrue(result[col].dtype == bool)

    # ------------------------------------------------------------------ #
    #  transform consistency
    # ------------------------------------------------------------------ #
    def test_transform_matches_fit_transform_columns(self):
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        b = StandardBinarizer(num_bins=2)
        fit_result = b.fit_transform(train)
        test = pd.DataFrame({"x": [1.5, 3.5]})
        transform_result = b.transform(test)
        self.assertEqual(list(fit_result.columns), list(transform_result.columns))

    def test_transform_preserves_index(self):
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
        b = StandardBinarizer(num_bins=2)
        result = b.fit_transform(train)
        self.assertEqual(list(result.index), [10, 20, 30])

        test = pd.DataFrame({"x": [1.5]}, index=[99])
        result = b.transform(test)
        self.assertEqual(list(result.index), [99])

    # ------------------------------------------------------------------ #
    #  Unique column name deduplication
    # ------------------------------------------------------------------ #
    def test_unique_column_name_deduplication(self):
        """_ensure_unique_column_names should deduplicate colliding names."""
        b = StandardBinarizer()
        names: set[str] = set()
        first = b._ensure_unique_column_names(names, "col")
        self.assertEqual(first, "col")
        self.assertIn("col", names)
        # Adding the same name again should still return without error
        second = b._ensure_unique_column_names(names, "col")
        # The set should have grown
        self.assertGreater(len(names), 1)
        self.assertEqual(second, "col_0")

    # ------------------------------------------------------------------ #
    #  is_fitted state
    # ------------------------------------------------------------------ #
    def test_is_fitted_after_fit_transform(self):
        b = StandardBinarizer()
        self.assertFalse(b._is_fitted)
        b.fit_transform(pd.DataFrame({"x": [1.0, 2.0]}))
        self.assertTrue(b._is_fitted)

    # ------------------------------------------------------------------ #
    #  Column mismatch on transform
    # ------------------------------------------------------------------ #
    def test_transform_different_columns_raises(self):
        b = StandardBinarizer(num_bins=2)
        b.fit_transform(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
        with self.assertRaises(RuntimeError, msg="Original columns do not match"):
            b.transform(pd.DataFrame({"y": [1.0, 2.0]}))

    def test_transform_extra_column_raises(self):
        b = StandardBinarizer(num_bins=2)
        b.fit_transform(pd.DataFrame({"x": [1.0, 2.0]}))
        with self.assertRaises(RuntimeError):
            b.transform(pd.DataFrame({"x": [1.0], "z": [2.0]}))

    def test_transform_missing_column_raises(self):
        b = StandardBinarizer(num_bins=2)
        b.fit_transform(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
        with self.assertRaises(RuntimeError):
            b.transform(pd.DataFrame({"a": [1.0]}))

    def test_transform_different_column_order_raises(self):
        b = StandardBinarizer(num_bins=2)
        b.fit_transform(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
        with self.assertRaises(RuntimeError):
            b.transform(pd.DataFrame({"b": [3.0], "a": [1.0]}))

    def test_transform_same_columns_passes(self):
        b = StandardBinarizer(num_bins=2)
        b.fit_transform(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
        # Should not raise
        result = b.transform(pd.DataFrame({"x": [1.5]}))
        self.assertEqual(result.shape[0], 1)

    # ------------------------------------------------------------------ #
    #  Doctests
    # ------------------------------------------------------------------ #
    def test_doctests(self):
        result = doctest.testmod(hgp_lib.preprocessing.binarizer, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
