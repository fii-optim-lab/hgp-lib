"""Tests for hgp_lib.utils.metrics."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np

import hgp_lib.utils.metrics
from hgp_lib.utils.metrics import (
    accepts_sample_weight,
    confusion_matrix,
    fast_f1_score,
    optimize_scorers_for_data,
    transform_duplicates_to_sample_weight,
)


class TestConfusionMatrix(unittest.TestCase):
    def test_basic(self):
        y_pred = np.array([True, True, False, False])
        y_true = np.array([True, False, True, False])
        tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
        self.assertEqual((tp, fp, fn, tn), (1, 1, 1, 1))

    def test_all_correct(self):
        y = np.array([True, False, True])
        tp, fp, fn, tn = confusion_matrix(y, y)
        self.assertEqual(tp, 2)
        self.assertEqual(tn, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

    def test_all_wrong(self):
        y_pred = np.array([True, False])
        y_true = np.array([False, True])
        tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
        self.assertEqual((tp, fp, fn, tn), (0, 1, 1, 0))

    def test_all_positive(self):
        y_pred = np.array([True, True, True])
        y_true = np.array([True, True, True])
        tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
        self.assertEqual((tp, fp, fn, tn), (3, 0, 0, 0))

    def test_all_negative(self):
        y_pred = np.array([False, False])
        y_true = np.array([False, False])
        tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
        self.assertEqual((tp, fp, fn, tn), (0, 0, 0, 2))

    def test_sum_equals_total(self):
        y_pred = np.array([True, False, True, True, False])
        y_true = np.array([False, False, True, False, True])
        tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
        self.assertEqual(tp + fp + fn + tn, len(y_pred))

    def test_with_sample_weight(self):
        y_pred = np.array([True, True, False])
        y_true = np.array([True, False, False])
        sw = np.array([2, 3, 1])
        tp, fp, fn, tn = confusion_matrix(y_true, y_pred, sample_weight=sw)
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 3)
        self.assertEqual(fn, 0)
        self.assertEqual(tn, 1)


class TestFastF1Score(unittest.TestCase):
    def test_perfect(self):
        y = np.array([True, False, True])
        self.assertAlmostEqual(fast_f1_score(y, y), 1.0)

    def test_known_value(self):
        y_pred = np.array([True, True, False, False])
        y_true = np.array([True, False, False, True])
        self.assertAlmostEqual(fast_f1_score(y_pred, y_true), 0.5)

    def test_no_predictions_no_labels(self):
        y_pred = np.array([False, False])
        y_true = np.array([False, False])
        self.assertAlmostEqual(fast_f1_score(y_pred, y_true), 1.0)

    def test_no_predictions_some_labels(self):
        y_pred = np.array([False, False])
        y_true = np.array([True, False])
        self.assertAlmostEqual(fast_f1_score(y_pred, y_true), 0.0)

    def test_some_predictions_no_labels(self):
        y_pred = np.array([True, False])
        y_true = np.array([False, False])
        self.assertAlmostEqual(fast_f1_score(y_pred, y_true), 0.0)

    def test_with_sample_weight(self):
        y_pred = np.array([True, True, False])
        y_true = np.array([True, False, True])
        sw = np.array([1, 1, 1])
        score = fast_f1_score(y_pred, y_true, sample_weight=sw)
        self.assertAlmostEqual(score, 0.5)


class TestAcceptsSampleWeight(unittest.TestCase):
    def test_with_param(self):
        def scorer(p, labels, sample_weight=None):
            return 0.0

        self.assertTrue(accepts_sample_weight(scorer))

    def test_with_param_non_inspectable(self):
        def scorer(p, labels, sample_weight=None):
            return 0.0

        with patch("inspect.signature", side_effect=TypeError("Cannot read signature")):
            self.assertTrue(accepts_sample_weight(scorer))

    def test_without_param(self):
        def scorer(p, labels):
            return 0.0

        self.assertFalse(accepts_sample_weight(scorer))

    def test_builtin(self):
        # builtins like len don't have inspectable signatures for sample_weight
        self.assertFalse(accepts_sample_weight(len))


class TestTransformDuplicates(unittest.TestCase):
    def test_removes_duplicates(self):
        data = np.array([[1, 0], [1, 0], [0, 1]])
        labels = np.array([1, 1, 0])
        ud, _ul, sw = transform_duplicates_to_sample_weight(data, labels)
        self.assertLess(len(ud), len(data))
        self.assertEqual(sw.sum(), len(data))

    def test_no_duplicates(self):
        data = np.array([[1, 0], [0, 1]])
        labels = np.array([1, 0])
        ud, _ul, sw = transform_duplicates_to_sample_weight(data, labels)
        self.assertEqual(len(ud), 2)
        np.testing.assert_array_equal(sw, [1, 1])

    def test_all_same(self):
        data = np.array([[1, 1], [1, 1], [1, 1]])
        labels = np.array([0, 0, 0])
        ud, _ul, sw = transform_duplicates_to_sample_weight(data, labels)
        self.assertEqual(len(ud), 1)
        self.assertEqual(sw[0], 3)

    def test_labels_preserved(self):
        data = np.array([[1, 0], [1, 0], [0, 1]])
        labels = np.array([1, 1, 0])
        ud, _ul, _sw = transform_duplicates_to_sample_weight(data, labels)
        # Each unique (data, label) pair should appear
        self.assertEqual(len(ud), 2)


class TestOptimizeScorersForData(unittest.TestCase):
    def _scorer_with_sw(self, p, labels, sample_weight=None):
        return float((p == labels).mean())

    def _scorer_without_sw(self, p, labels):
        return float((p == labels).mean())

    def test_optimizes_when_sw_supported(self):
        data = np.array([[1, 0], [1, 0], [0, 1]])
        labels = np.array([1, 1, 0])
        _opt_scorer, opt_data, _opt_labels = optimize_scorers_for_data(
            self._scorer_with_sw,
            data=data,
            labels=labels,
        )
        self.assertLessEqual(len(opt_data), len(data))

    def test_no_optimization_without_sw(self):
        data = np.array([[1, 0], [1, 0], [0, 1]])
        labels = np.array([1, 1, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _opt_scorer, opt_data, _opt_labels = optimize_scorers_for_data(
                self._scorer_without_sw,
                data=data,
                labels=labels,
            )
        # Data unchanged when scorer doesn't support sample_weight
        np.testing.assert_array_equal(opt_data, data)

    def test_warns_without_sw(self):
        # Clear the warned set to ensure warning fires
        hgp_lib.utils.metrics._warned_scorers.clear()
        data = np.array([[1, 0], [0, 1]])
        labels = np.array([1, 0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            optimize_scorers_for_data(
                self._scorer_without_sw,
                data=data,
                labels=labels,
            )
            self.assertTrue(any("sample_weight" in str(x.message) for x in w))

    def test_multiple_scorers(self):
        data = np.array([[1, 0], [0, 1]])
        labels = np.array([1, 0])
        s1, s2, _opt_data, _opt_labels = optimize_scorers_for_data(
            self._scorer_with_sw,
            self._scorer_with_sw,
            data=data,
            labels=labels,
        )
        self.assertTrue(callable(s1))
        self.assertTrue(callable(s2))

    def test_non_callable_raises(self):
        data = np.array([[1, 0]])
        labels = np.array([1])
        with self.assertRaises(TypeError):
            optimize_scorers_for_data(42, data=data, labels=labels)


if __name__ == "__main__":
    unittest.main()
