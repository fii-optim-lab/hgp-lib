import unittest

import numpy as np
import pandas as pd

from hgp_lib.preprocessing import StandardBinarizer
from hgp_lib.preprocessing.binning import (
    BinningStrategy,
    QuantileBinning,
    SupervisedTreeBinning,
)
from hgp_lib.preprocessing.warnings import (
    HighCardinalityWarning,
    StringColumnWarning,
    UnseenNaNWarning,
)
from hgp_lib.utils.warnings import _emitted_messages


class TestStandardBinarizer(unittest.TestCase):
    # Warnings deduplicate per process by message, so warning-asserting tests use
    # column names unique across the suite to stay independent of run order.
    def setUp(self):
        _emitted_messages.clear()

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #
    def test_num_bins_must_be_int(self):
        with self.assertRaises(TypeError):
            StandardBinarizer(num_bins=2.5)

    def test_num_bins_minimum(self):
        with self.assertRaises(ValueError):
            StandardBinarizer(num_bins=1)
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
        StandardBinarizer(precision=0)

    def test_numeric_binning_must_be_strategy(self):
        with self.assertRaises(TypeError):
            StandardBinarizer(numeric_binning=object())

    def test_fit_transform_requires_dataframe(self):
        b = StandardBinarizer()
        with self.assertRaises(TypeError):
            b.fit_transform(np.array([[1, 2]]))

    def test_transform_requires_dataframe(self):
        b = StandardBinarizer()
        b.fit_transform(pd.DataFrame({"x": [1.0, 2.0]}))
        with self.assertRaises(TypeError):
            b.transform(np.array([[1.0]]))

    def test_transform_before_fit_raises(self):
        b = StandardBinarizer()
        with self.assertRaises(ValueError):
            b.transform(pd.DataFrame({"x": [1.0]}))

    # ------------------------------------------------------------------ #
    #  Boolean columns
    # ------------------------------------------------------------------ #
    def test_bool_passthrough(self):
        df = pd.DataFrame({"flag": [True, False, True]})
        result = StandardBinarizer().fit_transform(df)
        self.assertEqual(list(result.columns), ["flag"])
        self.assertEqual(result["flag"].tolist(), [True, False, True])

    def test_bool_transform(self):
        b = StandardBinarizer()
        b.fit_transform(pd.DataFrame({"flag": [True, False]}))
        result = b.transform(pd.DataFrame({"flag": [False, True, True]}))
        self.assertEqual(result["flag"].tolist(), [False, True, True])

    # ------------------------------------------------------------------ #
    #  Categorical columns
    # ------------------------------------------------------------------ #
    def test_categorical_one_hot(self):
        df = pd.DataFrame({"color": pd.Categorical(["r", "g", "b", "r"])})
        result = StandardBinarizer().fit_transform(df)
        self.assertEqual(list(result.columns), ["color=r", "color=g", "color=b"])
        self.assertEqual(result["color=r"].tolist(), [True, False, False, True])
        self.assertEqual(result["color=g"].tolist(), [False, True, False, False])
        self.assertEqual(result["color=b"].tolist(), [False, False, True, False])

    def test_categorical_transform(self):
        b = StandardBinarizer()
        b.fit_transform(pd.DataFrame({"color": pd.Categorical(["r", "g", "b", "r"])}))
        result = b.transform(pd.DataFrame({"color": pd.Categorical(["g", "r"])}))
        self.assertEqual(result.shape[0], 2)
        self.assertTrue(result.iloc[0]["color=g"])

    def test_string_column_treated_as_categorical_with_warning(self):
        df = pd.DataFrame({"s_string": pd.array(["a", "a", "b"], dtype="string")})
        b = StandardBinarizer()
        with self.assertWarns(StringColumnWarning):
            result = b.fit_transform(df)
        self.assertEqual(list(result.columns), ["s_string=a", "s_string=b"])
        self.assertEqual(result["s_string=a"].tolist(), [True, True, False])
        self.assertEqual(result["s_string=b"].tolist(), [False, False, True])

    def test_object_column_treated_as_categorical_with_warning(self):
        df = pd.DataFrame({"s_object": ["a", "a", "b"]})
        b = StandardBinarizer()
        with self.assertWarns(StringColumnWarning):
            result = b.fit_transform(df)
        self.assertEqual(list(result.columns), ["s_object=a", "s_object=b"])

    def test_all_unique_categorical_is_skipped_with_warning(self):
        df = pd.DataFrame({"id_fit": pd.Categorical(["a", "b", "c"])})
        b = StandardBinarizer()
        with self.assertWarns(HighCardinalityWarning):
            result = b.fit_transform(df)
        self.assertEqual(result.shape, (3, 1))
        self.assertIn("id_fit", b._skipped_columns)

    def test_all_unique_categorical_skipped_on_transform(self):
        b = StandardBinarizer()
        with self.assertWarns(HighCardinalityWarning):
            b.fit_transform(pd.DataFrame({"id_tr": pd.Categorical(["a", "b", "c"])}))
        result = b.transform(pd.DataFrame({"id_tr": pd.Categorical(["a", "a", "a"])}))
        self.assertEqual(result.shape, (3, 1))

    # ------------------------------------------------------------------ #
    #  Numeric columns
    # ------------------------------------------------------------------ #
    def test_numeric_quantile_binning(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        result = StandardBinarizer(num_bins=2).fit_transform(df)
        for col in result.columns:
            self.assertTrue(result[col].dtype == bool)
        self.assertTrue((result.sum(axis=1) == 1).all())

    def test_numeric_exact_columns_and_values(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        result = StandardBinarizer(num_bins=2).fit_transform(df)
        self.assertEqual(list(result.columns), ["x < 2.500", "2.500 <= x"])
        self.assertEqual(result["x < 2.500"].tolist(), [True, True, False, False])
        self.assertEqual(result["2.500 <= x"].tolist(), [False, False, True, True])

    def test_numeric_column_strategy_override(self):
        df = pd.DataFrame({"a": range(20), "b": range(20)})
        b = StandardBinarizer(num_bins=3, column_strategy={"a": 2})
        result = b.fit_transform(df)
        self.assertEqual(len(b._numerical_bins["a"]) - 1, 2)
        self.assertEqual(len(b._numerical_bins["b"]) - 1, 3)
        self.assertEqual(result.shape, (20, 5))

    def test_numeric_constant_column(self):
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
        result = StandardBinarizer(num_bins=3).fit_transform(df)
        self.assertEqual(result.shape[1], 1)
        self.assertTrue(result.iloc[0, 0])

    def test_numeric_tree_binning(self):
        df = pd.DataFrame({"x": np.arange(100, dtype=float)})
        y = (df["x"] > 50).astype(int).values
        result = StandardBinarizer(num_bins=3).fit_transform(df, y=y)
        for col in result.columns:
            self.assertTrue(result[col].dtype == bool)
        self.assertTrue((result.sum(axis=1) == 1).all())

    def test_custom_numeric_binning_strategy(self):
        class MedianSplit(BinningStrategy):
            def compute_edges(self, values, y, n_bins):
                return np.array([-np.inf, float(np.median(values)), np.inf])

        b = StandardBinarizer(numeric_binning=MedianSplit())
        # Custom strategy is used even when labels are provided.
        result = b.fit_transform(
            pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}), y=np.array([0, 0, 1, 1])
        )
        self.assertEqual(result.shape, (4, 2))
        np.testing.assert_array_equal(b._numerical_bins["x"], [-np.inf, 2.5, np.inf])

    # ------------------------------------------------------------------ #
    #  NaN handling
    # ------------------------------------------------------------------ #
    def test_numeric_nan_creates_indicator_column(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0]})
        b = StandardBinarizer(num_bins=2)
        result = b.fit_transform(df)
        self.assertIn("x_is_NA", result.columns)
        self.assertEqual(result["x_is_NA"].tolist(), [False, True, False, False])
        self.assertIn("x", b._na_columns)
        # NaN row falls into no numeric bin.
        bin_cols = [c for c in result.columns if c != "x_is_NA"]
        self.assertEqual(result.loc[1, bin_cols].sum(), 0)

    def test_nan_indicator_preserved_on_transform(self):
        b = StandardBinarizer(num_bins=2)
        b.fit_transform(pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0]}))
        result = b.transform(pd.DataFrame({"x": [2.0, 3.0]}))
        # The NA indicator column exists even when new data has no NaN.
        self.assertIn("x_is_NA", result.columns)
        self.assertEqual(result["x_is_NA"].tolist(), [False, False])

    def test_unseen_nan_on_transform_warns(self):
        b = StandardBinarizer(num_bins=2)
        b.fit_transform(pd.DataFrame({"x_unseen": [1.0, 2.0, 3.0, 4.0]}))
        with self.assertWarns(UnseenNaNWarning):
            result = b.transform(pd.DataFrame({"x_unseen": [2.0, np.nan]}))
        # No indicator column created; NaN row is all-false.
        self.assertNotIn("x_unseen_is_NA", result.columns)
        self.assertEqual(result.iloc[1].sum(), 0)

    # ------------------------------------------------------------------ #
    #  Binning strategies
    # ------------------------------------------------------------------ #
    def test_quantile_constant_returns_single_bin(self):
        edges = QuantileBinning().compute_edges(np.array([5.0, 5.0, 5.0]), None, 3)
        np.testing.assert_array_equal(edges, [-np.inf, np.inf])

    def test_quantile_boundaries_and_sorted(self):
        edges = QuantileBinning().compute_edges(np.arange(100, dtype=float), None, 4)
        self.assertEqual(edges[0], -np.inf)
        self.assertEqual(edges[-1], np.inf)
        self.assertTrue(np.all(np.diff(edges) > 0))

    def test_quantile_deduplicates(self):
        X = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        edges = QuantileBinning().compute_edges(X, None, 5)
        self.assertEqual(len(edges), len(set(edges)))

    def test_tree_requires_labels(self):
        with self.assertRaises(ValueError):
            SupervisedTreeBinning().compute_edges(np.arange(10, dtype=float), None, 3)

    def test_tree_constant_returns_single_bin(self):
        edges = SupervisedTreeBinning().compute_edges(
            np.array([3.0, 3.0, 3.0]), np.array([0, 1, 0]), 3
        )
        np.testing.assert_array_equal(edges, [-np.inf, np.inf])

    def test_tree_boundaries_and_max_bins(self):
        X = np.arange(100, dtype=float)
        y = np.array([0, 1, 2, 3] * 25)
        edges = SupervisedTreeBinning().compute_edges(X, y, n_bins=4)
        self.assertEqual(edges[0], -np.inf)
        self.assertEqual(edges[-1], np.inf)
        self.assertLessEqual(len(edges) - 1, 4)

    # ------------------------------------------------------------------ #
    #  _ensure_unique_column_names
    # ------------------------------------------------------------------ #
    def test_ensure_unique_new_name(self):
        b = StandardBinarizer()
        names: set = set()
        self.assertEqual(b._ensure_unique_column_names(names, "col"), "col")
        self.assertIn("col", names)

    def test_ensure_unique_collision_adds_suffix(self):
        b = StandardBinarizer()
        names = {"col"}
        b._ensure_unique_column_names(names, "col")
        self.assertTrue(any(n.startswith("col_") for n in names))

    def test_ensure_unique_multiple_collisions(self):
        b = StandardBinarizer()
        names: set = set()
        b._ensure_unique_column_names(names, "x")
        b._ensure_unique_column_names(names, "x")
        b._ensure_unique_column_names(names, "x")
        self.assertEqual(len(names), 3)

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
        self.assertEqual(
            b._format_numeric_bin_name("y", 1.0, 2.0), "1.00000 <= y < 2.00000"
        )

    # ------------------------------------------------------------------ #
    #  Mixed types
    # ------------------------------------------------------------------ #
    def test_mixed_types(self):
        df = pd.DataFrame(
            {
                "flag": [True, False, True, False],
                "color": pd.Categorical(["a", "b", "a", "b"]),
                "val": [10.0, 20.0, 30.0, 40.0],
            }
        )
        result = StandardBinarizer(num_bins=2).fit_transform(df)
        self.assertGreaterEqual(result.shape[1], 4)
        for col in result.columns:
            self.assertTrue(result[col].dtype == bool)

    # ------------------------------------------------------------------ #
    #  transform consistency
    # ------------------------------------------------------------------ #
    def test_transform_matches_fit_transform_columns(self):
        b = StandardBinarizer(num_bins=2)
        fit_result = b.fit_transform(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}))
        transform_result = b.transform(pd.DataFrame({"x": [1.5, 3.5]}))
        self.assertEqual(list(fit_result.columns), list(transform_result.columns))

    def test_transform_preserves_index(self):
        b = StandardBinarizer(num_bins=2)
        result = b.fit_transform(
            pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
        )
        self.assertEqual(list(result.index), [10, 20, 30])
        result = b.transform(pd.DataFrame({"x": [1.5]}, index=[99]))
        self.assertEqual(list(result.index), [99])

    # ------------------------------------------------------------------ #
    #  dtype changes / column mismatch on transform
    # ------------------------------------------------------------------ #
    def test_transform_dtype_change_raises(self):
        b = StandardBinarizer()
        b.fit_transform(pd.DataFrame({"x": [True, False]}))
        with self.assertRaises(ValueError):
            b.transform(pd.DataFrame({"x": pd.array(["a", "b"], dtype="string")}))

    def test_transform_numeric_to_categorical_raises(self):
        b = StandardBinarizer(num_bins=2)
        b.fit_transform(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
        with self.assertRaises(ValueError):
            b.transform(pd.DataFrame({"x": pd.Categorical(["a", "b", "c"])}))

    def test_transform_different_columns_raises(self):
        b = StandardBinarizer(num_bins=2)
        b.fit_transform(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
        with self.assertRaises(RuntimeError):
            b.transform(pd.DataFrame({"y": [1.0, 2.0]}))

    def test_transform_different_column_order_raises(self):
        b = StandardBinarizer(num_bins=2)
        b.fit_transform(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
        with self.assertRaises(RuntimeError):
            b.transform(pd.DataFrame({"b": [3.0], "a": [1.0]}))

    def test_unsupported_dtype_fit_transform(self):
        b = StandardBinarizer()
        df = pd.DataFrame(
            {"t": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]}
        )
        with self.assertRaises(ValueError):
            b.fit_transform(df)

    # ------------------------------------------------------------------ #
    #  is_fitted state
    # ------------------------------------------------------------------ #
    def test_is_fitted_after_fit_transform(self):
        b = StandardBinarizer()
        self.assertFalse(b.is_fitted)
        b.fit_transform(pd.DataFrame({"x": [1.0, 2.0]}))
        self.assertTrue(b.is_fitted)


class TestSklearnBinarizer(unittest.TestCase):
    def test_transform_before_fit_raises(self):
        from sklearn.preprocessing import KBinsDiscretizer
        from hgp_lib.preprocessing import SklearnBinarizer

        b = SklearnBinarizer(
            KBinsDiscretizer(n_bins=2, encode="onehot-dense", strategy="uniform")
        )
        with self.assertRaises(ValueError):
            b.transform(pd.DataFrame({"x": [0.0, 1.0]}))

    def test_feature_names_positional_fallback(self):
        from hgp_lib.preprocessing import SklearnBinarizer

        # A transformer without get_feature_names_out triggers positional names.
        class _NoNamesTransformer:
            def fit_transform(self, X, y=None):
                return np.array([[0, 1], [1, 0]])

            def transform(self, X):
                return np.array([[0, 1], [1, 0]])

        b = SklearnBinarizer(_NoNamesTransformer())
        out = b.fit_transform(pd.DataFrame({"x": [0.0, 1.0]}))
        self.assertEqual(list(out.columns), ["feature_0", "feature_1"])
        self.assertTrue(out.to_numpy().dtype == bool)


if __name__ == "__main__":
    unittest.main()
