"""Unit tests for RunResult and ExperimentResult."""

import doctest
import unittest
from dataclasses import replace

import hgp_lib.metrics.results
from hgp_lib.metrics.core import GenerationMetrics
from hgp_lib.metrics.history import PopulationHistory
from hgp_lib.metrics.results import RunResult, ExperimentResult
from hgp_lib.rules import Literal


class _Helpers:
    """Shared factory methods for building test fixtures."""

    @staticmethod
    def make_gen(val_score=None):
        g = GenerationMetrics.from_population(
            best_idx=0,
            best_rule=Literal(value=0),
            train_scores=[0.8],
            complexities=[1],
            child_population_generation_metrics=[],
        )
        if val_score is not None:
            g = replace(g, val_score=val_score)
        return g

    @staticmethod
    def make_fold(rule_value=0, val_score=None, train_score=None, **kwargs):
        gens = []
        if val_score is not None or train_score is not None:
            g = _Helpers.make_gen(val_score=val_score)
            if train_score is not None:
                g = replace(g, train_scores=[train_score])
            gens = [g]
        defaults = dict(
            global_best_rule=Literal(value=rule_value),
            tp=3,
            fp=1,
            fn=2,
            tn=4,
            generations=gens,
        )
        defaults.update(kwargs)
        return PopulationHistory(**defaults)

    @staticmethod
    def make_run(
        run_id=0, seed=42, folds=None, best_fold_idx=0, test_score=0.8, **kwargs
    ):
        if folds is None:
            folds = [_Helpers.make_fold()]
        defaults = dict(
            run_id=run_id,
            seed=seed,
            best_fold_idx=best_fold_idx,
            folds=folds,
            test_score=test_score,
            test_tp=4,
            test_fp=1,
            test_fn=1,
            test_tn=4,
            feature_names={0: "col_a", 1: "col_b"},
        )
        defaults.update(kwargs)
        return RunResult(**defaults)


class TestRunResult(unittest.TestCase, _Helpers):
    # ------------------------------------------------------------------ #
    #  best_fold
    # ------------------------------------------------------------------ #
    def test_best_fold(self):
        f0 = self.make_fold(rule_value=0)
        f1 = self.make_fold(rule_value=1)
        run = self.make_run(folds=[f0, f1], best_fold_idx=1)
        self.assertIs(run.best_fold, f1)

    def test_best_fold_single(self):
        f = self.make_fold()
        run = self.make_run(folds=[f], best_fold_idx=0)
        self.assertIs(run.best_fold, f)

    # ------------------------------------------------------------------ #
    #  best_rule
    # ------------------------------------------------------------------ #
    def test_best_rule(self):
        f = self.make_fold(rule_value=7)
        run = self.make_run(folds=[f])
        self.assertEqual(str(run.best_rule), "7")

    def test_best_rule_from_correct_fold(self):
        f0 = self.make_fold(rule_value=0)
        f1 = self.make_fold(rule_value=9)
        run = self.make_run(folds=[f0, f1], best_fold_idx=1)
        self.assertEqual(str(run.best_rule), "9")

    # ------------------------------------------------------------------ #
    #  fold_val_scores
    # ------------------------------------------------------------------ #
    def test_fold_val_scores_all_have_val(self):
        f0 = self.make_fold(val_score=0.7)
        f1 = self.make_fold(val_score=0.9)
        run = self.make_run(folds=[f0, f1])
        self.assertEqual(run.fold_val_scores, [0.7, 0.9])

    def test_fold_val_scores_none_excluded(self):
        f0 = self.make_fold(val_score=0.7)
        f1 = self.make_fold()  # no val
        run = self.make_run(folds=[f0, f1])
        self.assertEqual(run.fold_val_scores, [0.7])

    def test_fold_val_scores_all_none(self):
        run = self.make_run(folds=[self.make_fold(), self.make_fold()])
        self.assertEqual(run.fold_val_scores, [])

    # ------------------------------------------------------------------ #
    #  mean_val_score
    # ------------------------------------------------------------------ #
    def test_mean_val_score(self):
        f0 = self.make_fold(val_score=0.6)
        f1 = self.make_fold(val_score=0.8)
        run = self.make_run(folds=[f0, f1])
        self.assertAlmostEqual(run.mean_val_score, 0.7)

    def test_mean_val_score_no_val(self):
        run = self.make_run(folds=[self.make_fold()])
        self.assertEqual(run.mean_val_score, 0.0)

    # ------------------------------------------------------------------ #
    #  Confusion matrix strings
    # ------------------------------------------------------------------ #
    def test_train_confusion_matrix(self):
        f = self.make_fold(tp=5, fp=2, fn=1, tn=7)
        run = self.make_run(folds=[f])
        self.assertEqual(run.train_confusion_matrix, "[TP: 5, FP: 2, FN: 1, TN: 7]")

    def test_val_confusion_matrix_with_val(self):
        f = self.make_fold(val_tp=3, val_fp=1, val_fn=0, val_tn=6)
        run = self.make_run(folds=[f])
        self.assertEqual(run.val_confusion_matrix, "[TP: 3, FP: 1, FN: 0, TN: 6]")

    def test_val_confusion_matrix_no_val(self):
        run = self.make_run(folds=[self.make_fold()])
        self.assertEqual(run.val_confusion_matrix, "[]")

    def test_test_confusion_matrix(self):
        run = self.make_run(test_tp=5, test_fp=2, test_fn=1, test_tn=7)
        self.assertEqual(run.test_confusion_matrix, "[TP: 5, FP: 2, FN: 1, TN: 7]")

    # ------------------------------------------------------------------ #
    #  fold_train_scores
    # ------------------------------------------------------------------ #
    def test_fold_train_scores(self):
        f0 = self.make_fold(train_score=0.7)
        f1 = self.make_fold(train_score=0.9)
        run = self.make_run(folds=[f0, f1])
        self.assertEqual(run.fold_train_scores, [0.7, 0.9])

    def test_fold_train_scores_empty_folds(self):
        run = self.make_run(folds=[self.make_fold()])  # no generations
        self.assertEqual(run.fold_train_scores, [])

    # ------------------------------------------------------------------ #
    #  mean_train_score
    # ------------------------------------------------------------------ #
    def test_mean_train_score(self):
        f0 = self.make_fold(train_score=0.6)
        f1 = self.make_fold(train_score=0.8)
        run = self.make_run(folds=[f0, f1])
        self.assertAlmostEqual(run.mean_train_score, 0.7)

    def test_mean_train_score_no_generations(self):
        run = self.make_run(folds=[self.make_fold()])
        self.assertEqual(run.mean_train_score, 0.0)

    # ------------------------------------------------------------------ #
    #  feature_names
    # ------------------------------------------------------------------ #
    def test_feature_names(self):
        run = self.make_run(feature_names={0: "age", 1: "income"})
        self.assertEqual(run.feature_names[0], "age")
        self.assertEqual(run.feature_names[1], "income")


class TestExperimentResult(unittest.TestCase, _Helpers):
    # ------------------------------------------------------------------ #
    #  best_run
    # ------------------------------------------------------------------ #
    def test_best_run_single(self):
        run = self.make_run(run_id=0)
        exp = ExperimentResult(runs=[run])
        self.assertIs(exp.best_run, run)

    def test_best_run_picks_highest_mean_val(self):
        r0 = self.make_run(run_id=0, folds=[self.make_fold(val_score=0.5)])
        r1 = self.make_run(run_id=1, folds=[self.make_fold(val_score=0.9)])
        exp = ExperimentResult(runs=[r0, r1])
        self.assertEqual(exp.best_run.run_id, 1)

    def test_best_run_no_val_uses_first(self):
        """When no folds have val scores, mean_val_score=0.0 for all; first run wins."""
        r0 = self.make_run(run_id=0)
        r1 = self.make_run(run_id=1)
        exp = ExperimentResult(runs=[r0, r1])
        self.assertEqual(exp.best_run.run_id, 0)

    def test_best_run_falls_back_to_train_score(self):
        """When no run has val scores, best_run should use mean_train_score."""
        r0 = self.make_run(run_id=0, folds=[self.make_fold(train_score=0.5)])
        r1 = self.make_run(run_id=1, folds=[self.make_fold(train_score=0.9)])
        exp = ExperimentResult(runs=[r0, r1])
        self.assertEqual(exp.best_run.run_id, 1)

    def test_best_run_prefers_val_over_train(self):
        """When val scores exist, best_run should use them even if train is higher."""
        r0 = self.make_run(
            run_id=0, folds=[self.make_fold(val_score=0.9, train_score=0.5)]
        )
        r1 = self.make_run(
            run_id=1, folds=[self.make_fold(val_score=0.3, train_score=0.99)]
        )
        exp = ExperimentResult(runs=[r0, r1])
        self.assertEqual(exp.best_run.run_id, 0)

    # ------------------------------------------------------------------ #
    #  best_rule
    # ------------------------------------------------------------------ #
    def test_best_rule(self):
        f = self.make_fold(rule_value=42)
        run = self.make_run(folds=[f])
        exp = ExperimentResult(runs=[run])
        self.assertEqual(str(exp.best_rule), "42")

    def test_best_rule_from_best_run(self):
        r0 = self.make_run(
            run_id=0, folds=[self.make_fold(rule_value=1, val_score=0.3)]
        )
        r1 = self.make_run(
            run_id=1, folds=[self.make_fold(rule_value=9, val_score=0.9)]
        )
        exp = ExperimentResult(runs=[r0, r1])
        self.assertEqual(str(exp.best_rule), "9")

    # ------------------------------------------------------------------ #
    #  test_scores
    # ------------------------------------------------------------------ #
    def test_test_scores(self):
        r0 = self.make_run(test_score=0.7)
        r1 = self.make_run(test_score=0.9)
        exp = ExperimentResult(runs=[r0, r1])
        self.assertEqual(exp.test_scores, [0.7, 0.9])

    def test_test_scores_single(self):
        exp = ExperimentResult(runs=[self.make_run(test_score=0.85)])
        self.assertEqual(exp.test_scores, [0.85])

    # ------------------------------------------------------------------ #
    #  Doctests
    # ------------------------------------------------------------------ #
    def test_doctests(self):
        result = doctest.testmod(hgp_lib.metrics.results, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
