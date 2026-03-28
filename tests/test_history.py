"""Unit tests for PopulationHistory."""

import doctest
import unittest
from dataclasses import replace

import hgp_lib.metrics.history
from hgp_lib.metrics.history import PopulationHistory
from hgp_lib.metrics.core import GenerationMetrics
from hgp_lib.rules import Literal


class TestPopulationHistory(unittest.TestCase):
    def _make_gen(self, val_score=None, train_score=0.8):
        g = GenerationMetrics.from_population(
            best_idx=0,
            best_rule=Literal(value=0),
            train_scores=[train_score],
            complexities=[1],
            child_population_generation_metrics=[],
        )
        if val_score is not None:
            g = replace(g, val_score=val_score)
        return g

    def _make_history(self, generations=None, **kwargs):
        defaults = dict(
            global_best_rule=Literal(value=0),
            tp=0,
            fp=0,
            fn=0,
            tn=0,
        )
        defaults.update(kwargs)
        if generations is not None:
            defaults["generations"] = generations
        return PopulationHistory(**defaults)

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def test_empty_history(self):
        ph = self._make_history()
        self.assertEqual(len(ph.generations), 0)

    def test_with_generations(self):
        ph = self._make_history(generations=[self._make_gen()])
        self.assertEqual(len(ph.generations), 1)

    def test_confusion_matrix_fields(self):
        ph = self._make_history(tp=5, fp=2, fn=1, tn=7)
        self.assertEqual(ph.tp, 5)
        self.assertEqual(ph.fp, 2)
        self.assertEqual(ph.fn, 1)
        self.assertEqual(ph.tn, 7)

    def test_val_confusion_matrix_defaults_none(self):
        ph = self._make_history()
        self.assertIsNone(ph.val_tp)
        self.assertIsNone(ph.val_fp)
        self.assertIsNone(ph.val_fn)
        self.assertIsNone(ph.val_tn)

    def test_val_confusion_matrix_set(self):
        ph = self._make_history(val_tp=3, val_fp=1, val_fn=0, val_tn=6)
        self.assertEqual(ph.val_tp, 3)
        self.assertEqual(ph.val_tn, 6)

    # ------------------------------------------------------------------ #
    #  best_val_score
    # ------------------------------------------------------------------ #
    def test_best_val_score_none_when_no_val(self):
        ph = self._make_history(generations=[self._make_gen()])
        self.assertIsNone(ph.best_val_score)

    def test_best_val_score_single(self):
        ph = self._make_history(generations=[self._make_gen(val_score=0.7)])
        self.assertAlmostEqual(ph.best_val_score, 0.7)

    def test_best_val_score_max(self):
        ph = self._make_history(
            generations=[
                self._make_gen(val_score=0.6),
                self._make_gen(val_score=0.9),
                self._make_gen(val_score=0.75),
            ]
        )
        self.assertAlmostEqual(ph.best_val_score, 0.9)

    def test_best_val_score_mixed_none(self):
        """Generations without val_score should be ignored."""
        ph = self._make_history(
            generations=[
                self._make_gen(),
                self._make_gen(val_score=0.5),
                self._make_gen(),
            ]
        )
        self.assertAlmostEqual(ph.best_val_score, 0.5)

    def test_best_val_score_all_none(self):
        ph = self._make_history(generations=[self._make_gen(), self._make_gen()])
        self.assertIsNone(ph.best_val_score)

    # ------------------------------------------------------------------ #
    #  global_best_rule
    # ------------------------------------------------------------------ #
    def test_global_best_rule(self):
        rule = Literal(value=5)
        ph = self._make_history(global_best_rule=rule)
        self.assertIs(ph.global_best_rule, rule)

    # ------------------------------------------------------------------ #
    #  Doctests
    # ------------------------------------------------------------------ #
    def test_doctests(self):
        result = doctest.testmod(hgp_lib.metrics.history, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
