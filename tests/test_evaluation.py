"""Tests for hgp_lib.evaluation."""

import os
import pickle
import subprocess
import sys
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from hgp_lib.evaluation import (
    Dataset,
    NumpyBackend,
    accepts_sample_weight,
    confusion_matrix,
    fast_accuracy_score,
    fast_f1_score,
    predict,
    score,
)
from hgp_lib.evaluation.dataset import take_weighted
from hgp_lib.evaluation.numpy.kernels import count_dtype
from hgp_lib.evaluation.numpy.predict import (
    EVALUATIONS,
    evaluate_block,
    evaluate_fused,
    evaluate_inplace,
)
from hgp_lib.evaluation.scorers import merges_rows
from hgp_lib.rules import And, Literal, Or, Rule
from hgp_lib.utils import warnings as hgp_warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def reference_evaluate(rule: Rule, data: np.ndarray) -> np.ndarray:
    """Straightforward evaluation, independent of the backend's algorithms."""
    if isinstance(rule, Literal):
        result = data[:, rule.value].copy()
    else:
        children = [reference_evaluate(subrule, data) for subrule in rule.subrules]
        reduce = np.logical_and if isinstance(rule, And) else np.logical_or
        result = reduce.reduce(children)
    return ~result if rule.negated else result


def random_rule(rng: np.random.Generator, num_features: int, depth: int) -> Rule:
    if depth == 0 or rng.random() < 0.3:
        return Literal(
            value=int(rng.integers(num_features)), negated=bool(rng.random() < 0.5)
        )
    operator = And if rng.random() < 0.5 else Or
    subrules = [
        random_rule(rng, num_features, depth - 1) for _ in range(rng.integers(2, 5))
    ]
    return operator(subrules, negated=bool(rng.random() < 0.5), copy_subrules=False)


def duplicated_data(
    rng: np.random.Generator, num_unique: int, num_rows: int, num_features: int
):
    """Rows drawn from a few unique rows, so merging has something to merge."""
    base = rng.integers(0, 2, size=(num_unique, num_features)).astype(bool)
    base_labels = rng.integers(0, 2, size=num_unique).astype(bool)
    pick = rng.integers(0, num_unique, size=num_rows)
    return base[pick], base_labels[pick]


def custom_weighted_f1(y_true, y_pred, sample_weight=None):
    """A custom scorer written only with operations NumPy arrays and tensors share."""
    if sample_weight is None:
        tp = float((y_true & y_pred).sum())
        predicted = float(y_pred.sum())
        positives = float(y_true.sum())
    else:
        tp = float((sample_weight * (y_true & y_pred)).sum())
        predicted = float((sample_weight * y_pred).sum())
        positives = float((sample_weight * y_true).sum())
    if positives == 0:
        return 1.0 if predicted == 0 else 0.0
    return 2 * tp / (predicted + positives)


def custom_plain_accuracy(y_true, y_pred):
    return fast_accuracy_score(y_true, y_pred)


class BatchedF1:
    """A custom scorer with a ``batched`` method, counting how it is called."""

    def __init__(self):
        self.calls = 0
        self.batched_calls = 0

    def __call__(self, y_true, y_pred, sample_weight=None):
        self.calls += 1
        return custom_weighted_f1(y_true, y_pred, sample_weight)

    def batched(self, y_true, y_preds, sample_weight=None):
        self.batched_calls += 1
        return [custom_weighted_f1(y_true, y_pred, sample_weight) for y_pred in y_preds]


class TestPredictAlgorithms(unittest.TestCase):
    def test_matches_reference_on_random_rules(self):
        rng = np.random.default_rng(0)
        data_c = rng.integers(0, 2, size=(200, 12)).astype(bool)
        data_f = np.asfortranarray(data_c)
        for _ in range(300):
            rule = random_rule(rng, 12, depth=4)
            expected = reference_evaluate(rule, data_c)
            for algorithm in (evaluate_block, evaluate_inplace, evaluate_fused):
                for data in (data_c, data_f):
                    np.testing.assert_array_equal(algorithm(rule, data), expected)

    def test_does_not_modify_data(self):
        rng = np.random.default_rng(1)
        data = rng.integers(0, 2, size=(50, 6)).astype(bool)
        data.setflags(write=False)
        for _ in range(100):
            rule = random_rule(rng, 6, depth=3)
            for algorithm in EVALUATIONS.values():
                algorithm(rule, data)

    def test_unsupported_rule_type(self):
        class Xor(Rule):
            pass

        rule = Xor([Literal(value=0), Literal(value=1)])
        data = np.ones((2, 2), dtype=bool)
        for algorithm in EVALUATIONS.values():
            with self.assertRaises(TypeError):
                algorithm(rule, data)

    def test_subclasses_of_and_and_or_are_evaluated(self):
        class MyAnd(And):
            pass

        rule = MyAnd([Literal(value=0), Literal(value=1, negated=True)])
        data = np.array([[True, False], [True, True]])
        for algorithm in EVALUATIONS.values():
            np.testing.assert_array_equal(algorithm(rule, data), [True, False])


class TestPublicApi(unittest.TestCase):
    def setUp(self):
        self.data = np.array(
            [[True, False], [True, True], [False, False], [False, True]]
        )
        self.labels = np.array([1, 1, 0, 0])

    def test_predict_matches_reference(self):
        rule = Or([Literal(value=0), Literal(value=1, negated=True)])
        expected = reference_evaluate(rule, self.data)
        np.testing.assert_array_equal(predict(rule, self.data), expected)
        for name in EVALUATIONS:
            backend = NumpyBackend(evaluation=name)
            np.testing.assert_array_equal(
                predict(rule, self.data, backend=backend), expected
            )

    def test_predict_rejects_non_binarized_data(self):
        rule = Literal(value=0)
        with self.assertRaises(TypeError):
            predict(rule, self.data.astype(int))
        with self.assertRaises(TypeError):
            predict(rule, self.data.tolist())
        with self.assertRaises(ValueError):
            predict(rule, self.data[:, 0])

    def test_score(self):
        rule = Literal(value=0)
        self.assertEqual(score(rule, self.data, self.labels), 1.0)
        self.assertEqual(score(rule, self.data, self.labels, fast_accuracy_score), 1.0)
        self.assertEqual(
            score(Literal(value=1), self.data, self.labels, fast_accuracy_score), 0.5
        )
        self.assertIsInstance(score(rule, self.data, self.labels), float)

    def test_score_has_the_backend_precision(self):
        rule = Literal(value=0)
        labels = np.array([1, 0, 0, 0])
        self.assertEqual(
            score(rule, self.data, labels),
            float(np.float32(fast_f1_score(labels, predict(rule, self.data)))),
        )
        backend = NumpyBackend(dtype=np.float64)
        self.assertEqual(score(rule, self.data, labels, backend=backend), 2 / 3)

    def test_score_with_custom_scorer(self):
        rule = Literal(value=1)
        self.assertEqual(
            score(rule, self.data, self.labels, custom_plain_accuracy), 0.5
        )

    def test_score_rejects_bad_input(self):
        rule = Literal(value=0)
        with self.assertRaises(TypeError):
            score(rule, self.data.astype(int), self.labels)
        with self.assertRaises(TypeError):
            score(rule, self.data, self.labels.astype(float))
        with self.assertRaises(ValueError):
            score(rule, self.data, np.array([0, 1, 2, 0]))
        with self.assertRaises(TypeError):
            score(rule, self.data, self.labels, None)

    def test_rule_evaluate_is_deprecated(self):
        hgp_warnings._emitted_messages.clear()
        rule = And([Literal(value=0), Literal(value=1)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = rule.evaluate(self.data)
        self.assertTrue(any(w.category is DeprecationWarning for w in caught))
        # The warning points at the caller, not at hgp_lib.
        self.assertEqual(caught[0].filename, __file__)
        np.testing.assert_array_equal(result, reference_evaluate(rule, self.data))


class TestScoringFunctions(unittest.TestCase):
    def test_f1_edge_cases(self):
        empty = np.array([False, False])
        some = np.array([True, False])
        self.assertEqual(fast_f1_score(empty, empty), 1.0)
        self.assertEqual(fast_f1_score(empty, some), 0.0)
        self.assertEqual(fast_f1_score(some, empty), 0.0)
        self.assertEqual(
            fast_f1_score(empty, empty, sample_weight=np.array([2, 1])), 1.0
        )
        self.assertIsInstance(fast_f1_score(some, some), float)

    def test_weights_equal_repeated_rows(self):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 2, size=30).astype(bool)
        y_pred = rng.integers(0, 2, size=30).astype(bool)
        weights = rng.integers(1, 5, size=30)
        repeated_true, repeated_pred = (
            np.repeat(y_true, weights),
            np.repeat(y_pred, weights),
        )
        for fn in (fast_f1_score, fast_accuracy_score):
            self.assertEqual(
                fn(y_true, y_pred, sample_weight=weights),
                fn(repeated_true, repeated_pred),
            )
        self.assertEqual(
            confusion_matrix(y_true, y_pred, sample_weight=weights),
            confusion_matrix(repeated_true, repeated_pred),
        )

    def test_confusion_matrix_sums_to_rows(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([True, True, False, False, True])
        tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
        self.assertEqual((tp, fp, fn, tn), (2, 1, 1, 1))
        self.assertTrue(all(isinstance(value, int) for value in (tp, fp, fn, tn)))

    def test_accepts_sample_weight(self):
        def with_weights(y_true, y_pred, sample_weight=None):
            return 0.0

        def with_kwargs(y_true, y_pred, **kwargs):
            return 0.0

        def fails_on_numpy(y_true, y_pred, **kwargs):
            raise RuntimeError("only works with tensors")

        self.assertTrue(accepts_sample_weight(with_weights))
        self.assertTrue(accepts_sample_weight(with_kwargs))
        self.assertFalse(accepts_sample_weight(custom_plain_accuracy))
        self.assertFalse(accepts_sample_weight(fails_on_numpy))
        self.assertFalse(accepts_sample_weight(len))
        with patch("inspect.signature", side_effect=TypeError("no signature")):
            self.assertTrue(accepts_sample_weight(with_weights))


class TestMergeRows(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(6)
        self.data, self.labels = duplicated_data(rng, 5, 40, 3)

    def test_merges_rows_follows_optimize_scorer(self):
        self.assertTrue(merges_rows(fast_f1_score, None))
        self.assertTrue(merges_rows(fast_accuracy_score, None))
        self.assertFalse(merges_rows(fast_f1_score, False))
        self.assertFalse(merges_rows(custom_weighted_f1, None))
        self.assertTrue(merges_rows(custom_weighted_f1, True))
        self.assertFalse(merges_rows(custom_weighted_f1, False))

    def test_bind_merges_rows_only_when_asked(self):
        dataset = Dataset(self.data, self.labels)
        merged = NumpyBackend().bind(dataset, merge_rows=True)
        self.assertIsNotNone(merged.dataset.sample_weight)
        self.assertEqual(merged.dataset.n_rows, 40)
        self.assertIsNone(NumpyBackend().bind(dataset).dataset.sample_weight)

    def test_bind_raises_for_a_scorer_without_sample_weight(self):
        dataset = Dataset(self.data, self.labels)
        with self.assertRaises(ValueError) as context:
            NumpyBackend().bind(dataset, custom_plain_accuracy, merge_rows=True)
        message = str(context.exception)
        self.assertIn("custom_plain_accuracy", message)
        self.assertIn("merge_rows=False", message)

    def test_bind_checks_the_batched_method(self):
        class BatchedWithoutWeights:
            def __call__(self, y_true, y_pred, sample_weight=None):
                return 0.0

            def batched(self, y_true, y_preds):
                return [0.0] * len(y_preds)

        dataset = Dataset(self.data, self.labels)
        with self.assertRaises(ValueError):
            NumpyBackend().bind(dataset, BatchedWithoutWeights(), merge_rows=True)

    def test_bind_rejects_weights_without_merging(self):
        merged = Dataset(self.data, self.labels).deduplicate()
        with self.assertRaises(ValueError):
            NumpyBackend().bind(merged, fast_f1_score)

    def test_bind_rejects_a_score_fn_that_is_not_callable(self):
        with self.assertRaises(TypeError):
            NumpyBackend().bind(Dataset(self.data, self.labels), None)


class TestDataset(unittest.TestCase):
    def test_deduplicate_keeps_total_weight(self):
        rng = np.random.default_rng(3)
        data, labels = duplicated_data(rng, 10, 200, 5)
        merged = Dataset(data, labels).deduplicate()
        self.assertLess(len(merged.labels), 200)
        self.assertEqual(merged.n_rows, 200)
        self.assertEqual(merged.n_features, 5)
        self.assertEqual(merged.sample_weight.dtype, np.int64)
        self.assertIs(merged.deduplicate(), merged)

    def test_deduplicate_adds_existing_weights(self):
        data = np.array([[True], [True], [False]])
        labels = np.array([True, True, False])
        merged = Dataset(data, labels, np.array([2, 3, 1])).deduplicate()
        self.assertEqual(merged.sample_weight.tolist(), [1, 5])
        self.assertEqual(merged.n_rows, 6)

    def test_deduplicate_keeps_rows_with_different_labels(self):
        data = np.array([[True], [True]])
        dataset = Dataset(data, np.array([True, False]))
        self.assertIs(dataset.deduplicate(), dataset)

    def test_take_unweighted_is_indexing(self):
        data = np.arange(12).reshape(6, 2) % 3 == 0
        labels = np.arange(6) % 2 == 0
        subset = Dataset(data, labels).take(np.array([4, 1]))
        np.testing.assert_array_equal(subset.data, data[[4, 1]])
        np.testing.assert_array_equal(subset.labels, labels[[4, 1]])
        self.assertIsNone(subset.sample_weight)

    def test_take_weighted_counts_original_rows(self):
        rng = np.random.default_rng(4)
        data, labels = duplicated_data(rng, 6, 100, 4)
        merged = Dataset(data, labels).deduplicate()
        # Original rows in the order the merged weights stand for them.
        expanded_data = np.repeat(merged.data, merged.sample_weight, axis=0)
        expanded_labels = np.repeat(merged.labels, merged.sample_weight)
        rows = np.sort(rng.choice(merged.n_rows, 40, replace=False))
        subset = merged.take(rows)
        self.assertEqual(subset.n_rows, 40)
        repeated = np.repeat(subset.data, subset.sample_weight, axis=0)
        np.testing.assert_array_equal(repeated, expanded_data[rows])
        np.testing.assert_array_equal(
            np.repeat(subset.labels, subset.sample_weight), expanded_labels[rows]
        )

    def test_take_drops_unit_weights(self):
        dataset = Dataset(
            np.eye(3, dtype=bool), np.array([True, False, True]), np.array([3, 1, 2])
        )
        self.assertIsNone(dataset.take(np.array([0, 3])).sample_weight)
        selected, counts = take_weighted(np.array([3, 1, 2]), np.array([0, 3]))
        self.assertEqual(selected.tolist(), [0, 1])
        self.assertIsNone(counts)

    def test_select_features_keeps_weights(self):
        weights = np.array([2, 1])
        dataset = Dataset(
            np.array([[True, False], [False, True]]), np.array([True, False]), weights
        )
        selected = dataset.select_features(np.array([1]))
        np.testing.assert_array_equal(selected.data, [[False], [True]])
        self.assertIs(selected.sample_weight, weights)


class TestNumpyBackend(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.data, self.labels = duplicated_data(rng, 25, 400, 8)
        self.rules = [random_rule(rng, 8, depth=3) for _ in range(60)]
        self.rules += [Literal(value=0), Literal(value=3, negated=True)]

    def expected_scores(self, fn, dtype=np.float32):
        scores = [
            fn(self.labels, reference_evaluate(rule, self.data)) for rule in self.rules
        ]
        return np.array(scores).astype(dtype)

    def backends(self, **options):
        for order in ("F", "C"):
            for evaluation in EVALUATIONS:
                yield NumpyBackend(order=order, evaluation=evaluation, **options)
                yield NumpyBackend(
                    order=order, evaluation=evaluation, batched=True, **options
                )

    def test_scores_on_merged_rows_equal_raw_scores(self):
        dataset = Dataset(self.data, self.labels)
        for fn in (fast_f1_score, fast_accuracy_score):
            for dtype in (np.float32, np.float64):
                expected = self.expected_scores(fn, dtype)
                for backend in self.backends(dtype=dtype):
                    for merge_rows in (True, False):
                        evaluator = backend.bind(dataset, fn, merge_rows=merge_rows)
                        self.assertEqual(
                            evaluator.dataset.sample_weight is not None, merge_rows
                        )
                        scores = evaluator.score(self.rules)
                        self.assertEqual(scores.dtype, dtype)
                        np.testing.assert_array_equal(scores, expected)

    def test_no_positive_labels(self):
        labels = np.zeros(len(self.labels), dtype=bool)
        rules = [
            Literal(value=0),
            And([Literal(value=0), Literal(value=0, negated=True)]),
        ]
        for backend in self.backends():
            for merge_rows in (True, False):
                evaluator = backend.bind(
                    Dataset(self.data, labels), merge_rows=merge_rows
                )
                self.assertEqual(evaluator.score(rules).tolist(), [0.0, 1.0])

    def test_batch_sizes(self):
        # 7 does not divide the number of rules, so the last batch is smaller.
        for fn in (fast_f1_score, fast_accuracy_score):
            expected = self.expected_scores(fn)
            for batch_size in (None, 1, 7, len(self.rules), 10 * len(self.rules)):
                for merge_rows in (True, False):
                    evaluator = NumpyBackend(batched=True, batch_size=batch_size).bind(
                        Dataset(self.data, self.labels), fn, merge_rows=merge_rows
                    )
                    np.testing.assert_array_equal(evaluator.score(self.rules), expected)
                    self.assertEqual(evaluator.score([]).shape, (0,))

    def test_confusion_matrix_on_merged_rows(self):
        merged = NumpyBackend().bind(Dataset(self.data, self.labels), merge_rows=True)
        for rule in self.rules:
            self.assertEqual(
                merged.confusion_matrix(rule),
                confusion_matrix(self.labels, reference_evaluate(rule, self.data)),
            )

    def test_custom_scorers(self):
        seen = []

        def spy(y_true, y_pred, sample_weight=None):
            seen.append(sample_weight)
            return custom_weighted_f1(y_true, y_pred, sample_weight)

        expected = self.expected_scores(fast_f1_score)
        for backend in self.backends():
            for merge_rows in (True, False):
                seen.clear()
                evaluator = backend.bind(
                    Dataset(self.data, self.labels), spy, merge_rows=merge_rows
                )
                np.testing.assert_allclose(
                    evaluator.score(self.rules), expected, rtol=1e-6
                )
                self.assertEqual(all(w is not None for w in seen), merge_rows)

    def test_custom_scorer_with_a_batched_method(self):
        expected = self.expected_scores(fast_f1_score)
        for merge_rows in (True, False):
            for batched in (True, False):
                scorer = BatchedF1()
                backend = NumpyBackend(
                    batched=batched, batch_size=7 if batched else None
                )
                evaluator = backend.bind(
                    Dataset(self.data, self.labels), scorer, merge_rows=merge_rows
                )
                np.testing.assert_allclose(
                    evaluator.score(self.rules), expected, rtol=1e-6
                )
                if batched:
                    self.assertEqual((scorer.calls, scorer.batched_calls), (0, 9))
                else:
                    self.assertEqual(scorer.batched_calls, 0)

    def test_subset(self):
        rng = np.random.default_rng(7)
        for merge_rows in (True, False):
            evaluator = NumpyBackend().bind(
                Dataset(self.data, self.labels), merge_rows=merge_rows
            )
            rows = rng.choice(evaluator.dataset.n_rows, 100, replace=False)
            subset = evaluator.subset(rows)
            self.assertEqual(subset.dataset.n_rows, 100)
            self.assertIs(subset.score_fn, evaluator.score_fn)
            taken = evaluator.dataset.take(rows)
            expected = NumpyBackend()._bind(taken, fast_f1_score, merge_rows)
            np.testing.assert_array_equal(
                subset.score(self.rules), expected.score(self.rules)
            )

    def test_evaluator_data_layout(self):
        dataset = Dataset(self.data, self.labels)
        self.assertTrue(NumpyBackend().bind(dataset).dataset.data.flags.f_contiguous)
        self.assertTrue(
            NumpyBackend(order="C").bind(dataset).dataset.data.flags.c_contiguous
        )

    def test_backend_conversions(self):
        backend = NumpyBackend()
        self.assertIs(backend.from_numpy(self.data), self.data)
        scores = np.array([0.5], dtype=np.float32)
        self.assertIs(backend.to_numpy(scores), scores)

    def test_options_are_validated(self):
        with self.assertRaises(ValueError):
            NumpyBackend(order="K")
        with self.assertRaises(ValueError) as context:
            NumpyBackend(evaluation="fast")
        self.assertIn("'block'", str(context.exception))
        with self.assertRaises(TypeError):
            NumpyBackend(batched="yes")
        for batch_size in (2.0, True, "8"):
            with self.assertRaises(TypeError):
                NumpyBackend(batched=True, batch_size=batch_size)
        with self.assertRaises(ValueError):
            NumpyBackend(batched=True, batch_size=0)
        with self.assertRaises(ValueError):
            NumpyBackend(batch_size=8)
        with self.assertRaises(ValueError):
            NumpyBackend(dtype=np.int32)
        with self.assertRaises(TypeError):
            NumpyBackend(dtype="not a dtype")
        self.assertEqual(NumpyBackend(batched=np.bool_(True)).batched, True)

    def test_dtype(self):
        self.assertEqual(NumpyBackend().dtype, np.dtype(np.float32))
        self.assertEqual(NumpyBackend(dtype="float64").dtype, np.dtype(np.float64))
        self.assertEqual(NumpyBackend(dtype=np.float64), NumpyBackend(dtype="float64"))

    def test_counts_switch_to_float64_when_float32_is_not_exact(self):
        float32 = np.dtype(np.float32)
        self.assertEqual(count_dtype(float32, 2**23), float32)
        self.assertEqual(count_dtype(float32, 2**23 + 1), np.dtype(np.float64))
        # 2**24 + 1 original rows: float32 counts would round to 2**24.
        data = np.array([[True], [True], [False]])
        labels = np.array([True, True, False])
        weights = np.array([2**24, 1, 1])
        evaluator = NumpyBackend()._bind(
            Dataset(data, labels, weights), fast_f1_score, True
        )
        self.assertEqual(
            evaluator.confusion_matrix(Literal(value=0)), (2**24 + 1, 0, 0, 1)
        )
        self.assertEqual(evaluator.score([Literal(value=0)]).tolist(), [1.0])

    def test_pickle_and_equality(self):
        backend = NumpyBackend(
            order="C", evaluation="block", batched=True, batch_size=8
        )
        self.assertEqual(pickle.loads(pickle.dumps(backend)), backend)
        self.assertEqual(NumpyBackend(), NumpyBackend())

    def test_pickle_evaluator(self):
        expected = self.expected_scores(fast_f1_score)
        for backend in (NumpyBackend(), NumpyBackend(batched=True, batch_size=7)):
            for merge_rows in (True, False):
                evaluator = backend.bind(
                    Dataset(self.data, self.labels), merge_rows=merge_rows
                )
                restored = pickle.loads(pickle.dumps(evaluator))
                np.testing.assert_array_equal(restored.score(self.rules), expected)


class TestLowMemoryEnvironmentVariable(unittest.TestCase):
    def run_python(self, code: str, **environment):
        env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src"), **environment}
        return subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

    def test_is_ignored_with_a_future_warning(self):
        code = (
            "from hgp_lib.evaluation import NumpyBackend; "
            "print(NumpyBackend().evaluation)"
        )
        for value in ("1", "0"):
            result = self.run_python(code, HGP_LOW_MEMORY=value)
            self.assertEqual(result.stdout.strip(), "fused")
            self.assertEqual(result.stderr.count("FutureWarning"), 1)
            self.assertIn("HGP_LOW_MEMORY", result.stderr)

    def test_no_warning_without_the_variable(self):
        environment = {k: v for k, v in os.environ.items() if k != "HGP_LOW_MEMORY"}
        with patch.dict(os.environ, environment, clear=True):
            result = self.run_python("import hgp_lib")
        self.assertNotIn("HGP_LOW_MEMORY", result.stderr)


if __name__ == "__main__":
    unittest.main()
