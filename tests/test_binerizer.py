import doctest
import unittest
import numpy as np
import pandas as pd

from hgp_lib.preprocessing import StandardBinarizer


class TestStandardBinarizer(unittest.TestCase):
    def setUp(self):
        """Set up test data used across test cases."""
        self.data = pd.DataFrame(
            {
                "bool_col": [True, False, True, False],
                "cat_col": pd.Categorical(["A", "B", "A", "C"]),
                "num_col": [1.0, 2.0, 3.0, 4.0],
            }
        )
        self.labels = np.array([0, 0, 1, 1])

    def test_initialization(self):
        """Test initialization parameters and validation."""
        with self.subTest("Testing valid initialization"):
            binarizer = StandardBinarizer(num_bins=5)
            self.assertEqual(binarizer.num_bins, 5)
            self.assertEqual(binarizer.column_strategy, {})

        with self.subTest("Testing invalid num_bins"):
            with self.assertRaises(ValueError):
                StandardBinarizer(num_bins=1)
            with self.assertRaises(ValueError):
                StandardBinarizer(num_bins="5")

        with self.subTest("Testing invalid column_strategy"):
            with self.assertRaises(ValueError):
                StandardBinarizer(column_strategy={"col": 1})
            with self.assertRaises(ValueError):
                StandardBinarizer(column_strategy="invalid")

    def test_fit_transform_unlabeled(self):
        """Test fit_transform without labels."""
        with self.subTest("Testing basic transformation"):
            binarizer = StandardBinarizer(num_bins=2)
            result = binarizer.fit_transform(self.data)

            # Check boolean column
            self.assertIn("bool_col", result.columns)
            self.assertEqual(result["bool_col"].dtype, bool)
            np.testing.assert_array_equal(result["bool_col"], self.data["bool_col"])

            # Check categorical columns
            expected_cat_cols = ["cat_col_A", "cat_col_B", "cat_col_C"]
            self.assertTrue(all(col in result.columns for col in expected_cat_cols))

            # Check numerical columns
            expected_num_cols = ["num_col_bin_0", "num_col_bin_1"]
            self.assertTrue(all(col in result.columns for col in expected_num_cols))

            # Check all columns are boolean
            self.assertTrue(all(result[col].dtype == bool for col in result.columns))

    def test_fit_transform_labeled(self):
        """Test fit_transform with labels."""
        with self.subTest("Testing supervised binning"):
            binarizer = StandardBinarizer(num_bins=2)
            result = binarizer.fit_transform(self.data, self.labels)

            expected_columns = {
                "bool_col",
                "cat_col_A",
                "cat_col_B",
                "cat_col_C",
                "num_col_bin_0",
                "num_col_bin_1",
            }
            self.assertEqual(set(result.columns), expected_columns)
            self.assertTrue(all(result[col].dtype == bool for col in result.columns))

    def test_transform(self):
        """Test transform on new data."""
        with self.subTest("Testing transform after fit"):
            binarizer = StandardBinarizer(num_bins=2)
            binarizer.fit_transform(self.data)

            new_data = pd.DataFrame(
                {
                    "bool_col": [True, False],
                    "cat_col": pd.Categorical(["A", "B"]),
                    "num_col": [1.5, 3.5],
                }
            )

            result = binarizer.transform(new_data)

            expected_columns = {
                "bool_col",
                "cat_col_A",
                "cat_col_B",
                "cat_col_C",
                "num_col_bin_0",
                "num_col_bin_1",
            }
            self.assertEqual(set(result.columns), expected_columns)

        with self.subTest("Testing transform without fit"):
            binarizer = StandardBinarizer()
            with self.assertRaises(ValueError):
                binarizer.transform(self.data)

    def test_column_strategy(self):
        """Test custom column binning strategy."""
        with self.subTest("Testing custom bins per column"):
            binarizer = StandardBinarizer(num_bins=2, column_strategy={"num_col": 3})
            result = binarizer.fit_transform(self.data)

            num_col_bins = sum(
                1 for col in result.columns if col.startswith("num_col_bin")
            )
            self.assertEqual(num_col_bins, 3)

    def test_edge_cases(self):
        """Test edge cases and special scenarios."""
        with self.subTest("Testing single value column"):
            data = pd.DataFrame({"num_col": [1.0, 1.0, 1.0, 1.0]})
            binarizer = StandardBinarizer(num_bins=3)
            result = binarizer.fit_transform(data)
            self.assertEqual(
                sum(1 for col in result.columns if col.startswith("num_col_bin")), 1
            )

        with self.subTest("Testing empty DataFrame"):
            data = pd.DataFrame(
                {"bool_col": [], "cat_col": pd.Categorical([]), "num_col": []}
            )
            binarizer = StandardBinarizer()
            result = binarizer.fit_transform(data)
            self.assertEqual(len(result), 0)

    def test_doctests(self):
        """Verify that all doctests pass."""
        import hgp_lib.preprocessing

        result = doctest.testmod(hgp_lib.preprocessing.binarizer, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
