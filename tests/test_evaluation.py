"""Tests for hgp_lib.evaluation."""

import os
import pickle
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from hgp_lib.evaluation import (
    Dataset,
    NumpyBackend,
    Scorer,
    accepts_sample_weight,
    confusion_matrix,
    fast_accuracy_score,
    fast_f1_score,
    predict,
    resolve_scorer,
    score,
)
from hgp_lib.evaluation.numpy.predict import evaluate, evaluate_low_memory
from hgp_lib.rules import And, Literal, Or, Rule
from hgp_lib.utils import warnings as hgp_warnings


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
    base_labels = rng.integers(0, 2, size=num_unique)
    pick = rng.integers(0, num_unique, size=num_rows)
    return base[pick], base_labels[pick]


def custom_weighted_f1(y_true, y_pred, sample_weight=None):
    return fast_f1_score(y_true.astype(bool), y_pred, sample_weight=sample_weight)


def custom_plain_accuracy(y_true, y_pred):
    return fast_accuracy_score(y_true, y_pred)


class TestPredictAlgorithms(unittest.TestCase):
    def test_matches_reference_on_random_rules(self):
        rng = np.random.default_rng(0)
        data_c = rng.integers(0, 2, size=(200, 12)).astype(bool)
        data_f = np.asfortranarray(data_c)
        for _ in range(300):
            rule = random_rule(rng, 12, depth=4)
            expected = reference_evaluate(rule, data_c)
            for algorithm in (evaluate, evaluate_low_memory):
                for data in (data_c, data_f):
                    np.testing.assert_array_equal(algorithm(rule, data), expected)

    def test_does_not_modify_data(self):
        rng = np.random.default_rng(1)
        data = rng.integers(0, 2, size=(50, 6)).astype(bool)
        original = data.copy()
        for _ in range(100):
            rule = random_rule(rng, 6, depth=3)
            evaluate(rule, data)
            evaluate_low_memory(rule, data)
        np.testing.assert_array_equal(data, original)

    def test_unsupported_rule_type(self):
        class Xor(Rule):
            pass

        rule = Xor([Literal(value=0), Literal(value=1)])
        data = np.ones((2, 2), dtype=bool)
        for algorithm in (evaluate, evaluate_low_memory):
            with self.assertRaises(TypeError):
                algorithm(rule, data)


class TestPublicApi(unittest.TestCase):
    def setUp(self):
        self.data = np.array(
            [[True, False], [True, True], [False, False], [False, True]]
        )
        self.labels = np.array([1, 1, 0, 0])

    def test_predict_matches_reference(self):
        rule = Or([Literal(value=0), Literal(value=1, negated=True)])
        np.testing.assert_array_equal(
            predict(rule, self.data), reference_evaluate(rule, self.data)
        )
        np.testing.assert_array_equal(
            predict(rule, self.data, backend=NumpyBackend(low_memory=True)),
            reference_evaluate(rule, self.data),
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

        self.assertTrue(accepts_sample_weight(with_weights))
        self.assertTrue(accepts_sample_weight(with_kwargs))
        self.assertFalse(accepts_sample_weight(custom_plain_accuracy))
        self.assertFalse(accepts_sample_weight(len))
        with patch("inspect.signature", side_effect=TypeError("no signature")):
            self.assertTrue(accepts_sample_weight(with_weights))


class TestResolveScorer(unittest.TestCase):
    def setUp(self):
        hgp_warnings._emitted_messages.clear()

    def test_defaults(self):
        self.assertEqual(resolve_scorer(), Scorer(fast_f1_score, merge_rows=True))
        self.assertTrue(resolve_scorer(fast_accuracy_score).merge_rows)
        self.assertFalse(resolve_scorer(fast_f1_score, optimize=False).merge_rows)

    def test_custom_scorers_are_not_merged_by_default(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertFalse(resolve_scorer(custom_weighted_f1).merge_rows)
            self.assertFalse(resolve_scorer(custom_plain_accuracy).merge_rows)
            self.assertFalse(
                resolve_scorer(custom_plain_accuracy, optimize=False).merge_rows
            )

    def test_custom_scorer_opt_in(self):
        self.assertTrue(resolve_scorer(custom_weighted_f1, optimize=True).merge_rows)

    def test_opt_in_without_sample_weight_warns_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = resolve_scorer(custom_plain_accuracy, optimize=True)
            second = resolve_scorer(custom_plain_accuracy, optimize=True)
        self.assertFalse(first.merge_rows or second.merge_rows)
        future = [w for w in caught if w.category is FutureWarning]
        self.assertEqual(len(future), 1)
        self.assertIn("custom_plain_accuracy", str(future[0].message))

    def test_not_callable(self):
        with self.assertRaises(TypeError):
            resolve_scorer(42)

    def test_scorer_passes_weights_only_when_given(self):
        seen = []

        def spy(y_true, y_pred, **kwargs):
            seen.append(kwargs)
            return 0.0

        scorer = Scorer(spy, merge_rows=True)
        scorer(np.array([True]), np.array([True]))
        scorer(np.array([True]), np.array([True]), np.array([2]))
        self.assertEqual(seen[0], {})
        self.assertEqual(list(seen[1]), ["sample_weight"])


class TestDataset(unittest.TestCase):
    def test_deduplicate_keeps_total_weight(self):
        rng = np.random.default_rng(3)
        data, labels = duplicated_data(rng, 10, 200, 5)
        merged = Dataset(data, labels).deduplicate()
        self.assertLess(len(merged.labels), 200)
        self.assertEqual(merged.n_rows, 200)
        self.assertEqual(merged.sample_weight.dtype, np.int64)
        self.assertIs(merged.deduplicate(), merged)

    def test_deduplicate_adds_existing_weights(self):
        data = np.array([[True], [True], [False]])
        merged = Dataset(data, np.array([1, 1, 0]), np.array([2, 3, 1])).deduplicate()
        self.assertEqual(merged.sample_weight.tolist(), [1, 5])
        self.assertEqual(merged.n_rows, 6)

    def test_deduplicate_keeps_rows_with_different_labels(self):
        data = np.array([[True], [True]])
        dataset = Dataset(data, np.array([1, 0]))
        self.assertIs(dataset.deduplicate(), dataset)

    def test_take_unweighted_is_indexing(self):
        data = np.arange(12).reshape(6, 2) % 3 == 0
        labels = np.arange(6)
        subset = Dataset(data, labels).take(np.array([4, 1]))
        np.testing.assert_array_equal(subset.data, data[[4, 1]])
        np.testing.assert_array_equal(subset.labels, [4, 1])
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
            np.eye(3, dtype=bool), np.array([1, 0, 1]), np.array([3, 1, 2])
        )
        self.assertIsNone(dataset.take(np.array([0, 3])).sample_weight)

    def test_select_features_keeps_weights(self):
        weights = np.array([2, 1])
        dataset = Dataset(
            np.array([[True, False], [False, True]]), np.array([1, 0]), weights
        )
        selected = dataset.select_features(np.array([1]))
        np.testing.assert_array_equal(selected.data, [[False], [True]])
        self.assertIs(selected.sample_weight, weights)


class TestNumpyBackend(unittest.TestCase):
    def setUp(self):
        hgp_warnings._emitted_messages.clear()
        rng = np.random.default_rng(5)
        self.data, self.labels = duplicated_data(rng, 25, 400, 8)
        self.rules = [random_rule(rng, 8, depth=3) for _ in range(60)]
        self.rules += [Literal(value=0), Literal(value=3, negated=True)]

    def expected_scores(self, fn):
        return np.array(
            [
                fn(self.labels, reference_evaluate(rule, self.data))
                for rule in self.rules
            ]
        )

    def test_scores_on_merged_rows_equal_raw_scores(self):
        for fn in (fast_f1_score, fast_accuracy_score):
            expected = self.expected_scores(fn)
            for backend in (
                NumpyBackend(),
                NumpyBackend(order="C"),
                NumpyBackend(low_memory=True),
                NumpyBackend(batched=True),
            ):
                for merge_rows in (True, False):
                    evaluator = backend.bind(
                        Dataset(self.data, self.labels), Scorer(fn, merge_rows)
                    )
                    self.assertEqual(
                        evaluator.dataset.sample_weight is not None, merge_rows
                    )
                    np.testing.assert_array_equal(evaluator.score(self.rules), expected)

    def test_no_positive_labels(self):
        labels = np.zeros(len(self.labels), dtype=int)
        rules = [
            Literal(value=0),
            And([Literal(value=0), Literal(value=0, negated=True)]),
        ]
        for batched in (False, True):
            for merge_rows in (True, False):
                evaluator = NumpyBackend(batched=batched).bind(
                    Dataset(self.data, labels), Scorer(fast_f1_score, merge_rows)
                )
                self.assertEqual(evaluator.score(rules).tolist(), [0.0, 1.0])

    def test_batch_sizes(self):
        # 7 does not divide the number of rules, so the last batch is smaller.
        for fn in (fast_f1_score, fast_accuracy_score):
            expected = self.expected_scores(fn)
            for batch_size in (None, 1, 7, len(self.rules), 10 * len(self.rules)):
                for merge_rows in (True, False):
                    evaluator = NumpyBackend(batched=True, batch_size=batch_size).bind(
                        Dataset(self.data, self.labels), Scorer(fn, merge_rows)
                    )
                    np.testing.assert_array_equal(evaluator.score(self.rules), expected)
                    self.assertEqual(evaluator.score([]).shape, (0,))

    def test_confusion_matrix_on_merged_rows(self):
        merged = NumpyBackend().bind(Dataset(self.data, self.labels), resolve_scorer())
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
        for merge_rows in (True, False):
            seen.clear()
            evaluator = NumpyBackend(batched=True).bind(
                Dataset(self.data, self.labels), Scorer(spy, merge_rows)
            )
            np.testing.assert_array_equal(evaluator.score(self.rules), expected)
            self.assertEqual(all(w is not None for w in seen), merge_rows)
            # Custom scorers receive the user's labels, not a converted copy.
            self.assertEqual(evaluator.dataset.labels.dtype, self.labels.dtype)

    def test_evaluator_data_layout(self):
        dataset = Dataset(self.data, self.labels)
        self.assertTrue(
            NumpyBackend()
            .bind(dataset, resolve_scorer())
            .dataset.data.flags.f_contiguous
        )
        self.assertTrue(
            NumpyBackend(order="C")
            .bind(dataset, resolve_scorer())
            .dataset.data.flags.c_contiguous
        )

    def test_bind_rejects_weights_for_scorer_without_merging(self):
        merged = Dataset(self.data, self.labels).deduplicate()
        with self.assertRaises(ValueError):
            NumpyBackend().bind(merged, Scorer(fast_f1_score, merge_rows=False))

    def test_options_are_validated(self):
        with self.assertRaises(ValueError):
            NumpyBackend(order="K")
        with self.assertRaises(TypeError):
            NumpyBackend(low_memory=1)
        with self.assertRaises(TypeError):
            NumpyBackend(batched="yes")
        for batch_size in (2.0, True, "8"):
            with self.assertRaises(TypeError):
                NumpyBackend(batched=True, batch_size=batch_size)
        with self.assertRaises(ValueError):
            NumpyBackend(batched=True, batch_size=0)
        with self.assertRaises(ValueError):
            NumpyBackend(batch_size=8)

    def test_pickle_and_equality(self):
        backend = NumpyBackend(order="C", low_memory=True, batched=True, batch_size=8)
        self.assertEqual(pickle.loads(pickle.dumps(backend)), backend)
        self.assertEqual(NumpyBackend(), NumpyBackend())

    def test_pickle_evaluator(self):
        expected = self.expected_scores(fast_f1_score)
        for backend in (NumpyBackend(), NumpyBackend(batched=True, batch_size=7)):
            for merge_rows in (True, False):
                evaluator = backend.bind(
                    Dataset(self.data, self.labels), Scorer(fast_f1_score, merge_rows)
                )
                restored = pickle.loads(pickle.dumps(evaluator))
                np.testing.assert_array_equal(restored.score(self.rules), expected)

    def test_low_memory_default(self):
        with patch.dict(os.environ, clear=False) as environment:
            environment.pop("HGP_LOW_MEMORY", None)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                self.assertTrue(NumpyBackend().low_memory)

    def test_low_memory_environment_variable_is_deprecated(self):
        for value, expected in (("1", True), ("0", False)):
            hgp_warnings._emitted_messages.clear()
            with patch.dict(os.environ, {"HGP_LOW_MEMORY": value}):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    backend = NumpyBackend()
            self.assertEqual(backend.low_memory, expected)
            self.assertTrue(any(w.category is FutureWarning for w in caught))
        # An explicit option wins over the environment variable.
        with patch.dict(os.environ, {"HGP_LOW_MEMORY": "0"}):
            self.assertTrue(NumpyBackend(low_memory=True).low_memory)


if __name__ == "__main__":
    unittest.main()
