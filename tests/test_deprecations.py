"""Deprecated 1.x import paths still work in 2.x, with a warning at the user's line."""

import importlib
import unittest
import warnings

import numpy as np
import pandas as pd

import hgp_lib.evaluation
import hgp_lib.results
import hgp_lib.rules
from hgp_lib import BooleanRuleClassifier
from hgp_lib.configs import (
    BenchmarkerConfig,
    BooleanGPConfig,
    TrainerConfig,
    validate_benchmarker_config,
    validate_trainer_config,
)
from hgp_lib.utils import warnings as hgp_warnings


class TestMovedNames(unittest.TestCase):
    def setUp(self):
        hgp_warnings._emitted_messages.clear()

    def assert_moved(self, statement, expected):
        namespace = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec(compile(statement, __file__, "exec"), namespace)
        deprecations = [w for w in caught if w.category is DeprecationWarning]
        self.assertEqual(len(deprecations), 1, statement)
        self.assertEqual(deprecations[0].filename, __file__)
        name = statement.split()[-1]
        self.assertIs(namespace[name], expected)

    def test_metrics_package_is_results(self):
        self.assert_moved(
            "from hgp_lib.metrics import RunResult", hgp_lib.results.RunResult
        )
        self.assert_moved(
            "from hgp_lib.metrics import PopulationHistory",
            hgp_lib.results.PopulationHistory,
        )

    def test_utils_metrics_is_evaluation(self):
        self.assert_moved(
            "from hgp_lib.utils.metrics import fast_f1_score",
            hgp_lib.evaluation.fast_f1_score,
        )
        self.assert_moved(
            "from hgp_lib.utils.metrics import confusion_matrix",
            hgp_lib.evaluation.confusion_matrix,
        )

    def test_removed_helpers_explain_the_replacement(self):
        module = importlib.import_module("hgp_lib.utils.metrics")
        with self.assertRaises(AttributeError) as context:
            module.optimize_scorers_for_data  # noqa: B018
        self.assertIn("EvaluationBackend.bind", str(context.exception))
        with self.assertRaises(ImportError):
            exec("from hgp_lib.utils.metrics import SampleWeightScorer", {})

    def test_complexity_check_moved_to_rules(self):
        self.assert_moved(
            "from hgp_lib.utils import ComplexityCheck", hgp_lib.rules.ComplexityCheck
        )
        self.assert_moved(
            "from hgp_lib.utils.validation import ComplexityCheck",
            hgp_lib.rules.ComplexityCheck,
        )

    def test_unknown_names_still_fail(self):
        with self.assertRaises(ImportError):
            exec("from hgp_lib.metrics import Nothing", {})
        with self.assertRaises(ImportError):
            exec("from hgp_lib.utils import Nothing", {})


class TestDefaults(unittest.TestCase):
    def test_trainer_config_defaults(self):
        config = TrainerConfig()
        self.assertIsInstance(config.gp_config, BooleanGPConfig)
        self.assertEqual(config.num_epochs, 1000)
        validate_trainer_config(config, require_data=False)
        # Each config gets its own nested defaults.
        self.assertIsNot(TrainerConfig().gp_config, config.gp_config)

    def test_benchmarker_config_needs_only_data(self):
        data = pd.DataFrame({"a": [True, False] * 4, "b": [False, True] * 4})
        config = BenchmarkerConfig(data=data, labels=np.array([1, 0] * 4))
        self.assertIsInstance(config.trainer_config, TrainerConfig)
        validate_benchmarker_config(config)

    def test_classifier_without_config(self):
        classifier = BooleanRuleClassifier()
        self.assertEqual(classifier.trainer_config.num_epochs, 1000)

    def test_positional_trainer_config_still_works(self):
        config = TrainerConfig(BooleanGPConfig(), 10)
        self.assertEqual(config.num_epochs, 10)


if __name__ == "__main__":
    unittest.main()
