"""Tests for BooleanRuleClassifier: the binarize-and-train pipeline."""

import unittest

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from hgp_lib import BooleanRuleClassifier
from hgp_lib.configs import BooleanGPConfig, TrainerConfig
from hgp_lib.metrics import PopulationHistory
from hgp_lib.preprocessing import SklearnBinarizer, StandardBinarizer
from hgp_lib.rules import Rule
from hgp_lib.evaluation.scorer import fast_f1_score


class TestBooleanRuleClassifier(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.data = pd.DataFrame(
            {
                "num": rng.normal(size=40),
                "flag": rng.integers(0, 2, size=40).astype(bool),
            }
        )
        self.labels = rng.integers(0, 2, size=40)

    def _make_config(self, **kwargs):
        defaults = {
            "gp_config": BooleanGPConfig(score_fn=fast_f1_score),
            "num_epochs": 10,
            "progress_bar": False,
        }
        defaults.update(kwargs)
        return TrainerConfig(**defaults)

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def test_default_binarizer_is_standard(self):
        clf = BooleanRuleClassifier(self._make_config())
        self.assertIsInstance(clf.binarizer, StandardBinarizer)

    def test_custom_binarizer_is_used(self):
        from sklearn.preprocessing import KBinsDiscretizer

        binarizer = SklearnBinarizer(
            KBinsDiscretizer(n_bins=3, encode="onehot-dense", strategy="uniform")
        )
        clf = BooleanRuleClassifier(self._make_config(), binarizer=binarizer)
        self.assertIs(clf.binarizer, binarizer)

    def test_fitted_binarizer_rejected(self):
        binarizer = StandardBinarizer()
        binarizer.fit_transform(self.data)
        with self.assertRaises(ValueError):
            BooleanRuleClassifier(self._make_config(), binarizer=binarizer)

    # ------------------------------------------------------------------ #
    #  fit / predict
    # ------------------------------------------------------------------ #
    def test_fit_returns_history(self):
        clf = BooleanRuleClassifier(self._make_config())
        history = clf.fit(self.data, self.labels)
        self.assertIsInstance(history, PopulationHistory)
        self.assertIsInstance(history.global_best_rule, Rule)

    def test_predict_shape_and_dtype(self):
        clf = BooleanRuleClassifier(self._make_config())
        clf.fit(self.data, self.labels)
        preds = clf.predict(self.data)
        self.assertEqual(preds.shape, (len(self.data),))
        self.assertTrue(preds.dtype == bool)

    def test_predict_on_new_rows(self):
        clf = BooleanRuleClassifier(self._make_config())
        clf.fit(self.data, self.labels)
        preds = clf.predict(self.data.iloc[:5])
        self.assertEqual(preds.shape, (5,))

    def test_fit_with_validation_data_tracks_val_score(self):
        clf = BooleanRuleClassifier(self._make_config(val_every=1))
        history = clf.fit(
            self.data.iloc[:30],
            self.labels[:30],
            self.data.iloc[30:],
            self.labels[30:],
        )
        self.assertIsNotNone(history.best_val_score)
        self.assertEqual(clf.predict(self.data.iloc[30:]).shape, (10,))

    def test_fit_requires_both_val_args(self):
        clf = BooleanRuleClassifier(self._make_config())
        with self.assertRaises(ValueError):
            clf.fit(self.data, self.labels, X_val=self.data)
        with self.assertRaises(ValueError):
            clf.fit(self.data, self.labels, y_val=self.labels)

    def test_fit_requires_dataframe(self):
        clf = BooleanRuleClassifier(self._make_config())
        with self.assertRaises(TypeError):
            clf.fit(self.data.to_numpy(), self.labels)

    def test_predict_requires_dataframe(self):
        clf = BooleanRuleClassifier(self._make_config())
        clf.fit(self.data, self.labels)
        with self.assertRaises(TypeError):
            clf.predict(self.data.to_numpy())

    # ------------------------------------------------------------------ #
    #  Introspection
    # ------------------------------------------------------------------ #
    def test_feature_names_match_binarizer(self):
        clf = BooleanRuleClassifier(self._make_config())
        clf.fit(self.data, self.labels)
        self.assertEqual(clf.feature_names, clf.binarizer.get_feature_names_out())
        self.assertTrue(all(isinstance(name, str) for name in clf.feature_names))

    def test_rule_is_global_best(self):
        clf = BooleanRuleClassifier(self._make_config())
        history = clf.fit(self.data, self.labels)
        self.assertIs(clf.rule, history.global_best_rule)

    def test_format_rule_is_readable_string(self):
        clf = BooleanRuleClassifier(self._make_config())
        clf.fit(self.data, self.labels)
        formatted = clf.format_rule()
        self.assertIsInstance(formatted, str)
        self.assertGreater(len(formatted), 0)

    # ------------------------------------------------------------------ #
    #  Guardrails before fit
    # ------------------------------------------------------------------ #
    def test_predict_before_fit_raises(self):
        clf = BooleanRuleClassifier(self._make_config())
        with self.assertRaises(NotFittedError):
            clf.predict(self.data)

    def test_rule_before_fit_raises(self):
        clf = BooleanRuleClassifier(self._make_config())
        with self.assertRaises(NotFittedError):
            _ = clf.rule

    def test_feature_names_before_fit_raises(self):
        clf = BooleanRuleClassifier(self._make_config())
        with self.assertRaises(NotFittedError):
            _ = clf.feature_names


if __name__ == "__main__":
    unittest.main()
