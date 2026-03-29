"""Tests for hgp_lib.preprocessing.utils.load_data using temporary files."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from hgp_lib.preprocessing.utils import load_data


class TestLoadData(unittest.TestCase):
    def _write_csv(self, df: pd.DataFrame, tmpdir: str, name: str = "data.csv") -> str:
        path = os.path.join(tmpdir, name)
        df.to_csv(path, index=False)
        return path

    def _write_hdf(self, df: pd.DataFrame, tmpdir: str, name: str = "data.hdf") -> str:
        path = os.path.join(tmpdir, name)
        df.to_hdf(path, key="data", mode="w")
        return path

    def _sample_df(self):
        return pd.DataFrame(
            {
                "feat_a": [1.0, 2.0, 3.0, 4.0],
                "feat_b": [True, False, True, False],
                "target": [1, 0, 1, 0],
            }
        )

    # ------------------------------------------------------------------ #
    #  CSV
    # ------------------------------------------------------------------ #
    def test_load_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_csv(self._sample_df(), tmpdir)
            data, labels = load_data(path)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertIsInstance(labels, np.ndarray)
            self.assertEqual(len(data), 4)
            self.assertEqual(len(labels), 4)

    def test_csv_target_not_in_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_csv(self._sample_df(), tmpdir)
            data, _ = load_data(path)
            self.assertNotIn("target", data.columns)

    def test_csv_labels_are_bool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_csv(self._sample_df(), tmpdir)
            _, labels = load_data(path)
            self.assertEqual(labels.dtype, bool)

    def test_csv_labels_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_csv(self._sample_df(), tmpdir)
            _, labels = load_data(path)
            np.testing.assert_array_equal(labels, [True, False, True, False])

    def test_csv_feature_columns_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_csv(self._sample_df(), tmpdir)
            data, _ = load_data(path)
            self.assertEqual(list(data.columns), ["feat_a", "feat_b"])

    # ------------------------------------------------------------------ #
    #  HDF
    # ------------------------------------------------------------------ #
    def test_load_hdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_hdf(self._sample_df(), tmpdir)
            data, labels = load_data(path)
            self.assertEqual(len(data), 4)
            self.assertEqual(len(labels), 4)
            self.assertNotIn("target", data.columns)

    def test_hdf_labels_are_bool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_hdf(self._sample_df(), tmpdir)
            _, labels = load_data(path)
            self.assertEqual(labels.dtype, bool)

    # ------------------------------------------------------------------ #
    #  Error cases
    # ------------------------------------------------------------------ #
    def test_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nonexistent.csv")
            with self.assertRaises(FileNotFoundError):
                load_data(path)

    def test_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.txt")
            with open(path, "w") as f:
                f.write("some text")
            with self.assertRaises(ValueError):
                load_data(path)

    def test_missing_target_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            path = self._write_csv(df, tmpdir)
            with self.assertRaises(RuntimeError):
                load_data(path)

    # ------------------------------------------------------------------ #
    #  Edge cases
    # ------------------------------------------------------------------ #
    def test_target_only(self):
        """DataFrame with only a target column should return empty features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"target": [1, 0, 1]})
            path = self._write_csv(df, tmpdir)
            data, labels = load_data(path)
            self.assertEqual(data.shape, (3, 0))
            self.assertEqual(len(labels), 3)

    def test_single_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"x": [1.0], "target": [1]})
            path = self._write_csv(df, tmpdir)
            data, labels = load_data(path)
            self.assertEqual(len(data), 1)
            self.assertEqual(len(labels), 1)


if __name__ == "__main__":
    unittest.main()
