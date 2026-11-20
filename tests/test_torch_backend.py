"""Tests for hgp_lib.evaluation.torch. Skipped when PyTorch is not installed."""

import pickle
import random
import unittest
import warnings

import numpy as np

from hgp_lib.algorithms import BooleanGP
from hgp_lib.configs import BooleanGPConfig
from hgp_lib.evaluation import (
    Dataset,
    NumpyBackend,
    confusion_matrix,
    fast_accuracy_score,
    fast_f1_score,
    predict,
    score,
)
from hgp_lib.populations import (
    BestLiteralStrategy,
    CombinedSamplingStrategy,
    PopulationGeneratorFactory,
)
from hgp_lib.rules import And, Literal, Rule

from test_evaluation import (
    BatchedF1,
    custom_weighted_f1,
    duplicated_data,
    random_rule,
    reference_evaluate,
)

try:
    import torch

    from hgp_lib.evaluation.torch import TorchBackend, TorchDataset
    from hgp_lib.evaluation.torch.kernels import count_dtype
    from hgp_lib.evaluation.torch.predict import EVALUATIONS
except ImportError:  # pragma: no cover - only without PyTorch
    torch = None


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


class LiteralPairs(PopulationGeneratorFactory):
    """Populations that start from the best literals of random row subsets."""

    def create_strategies(self, num_literals, evaluator):
        return [BestLiteralStrategy(num_literals, evaluator, sample_size=0.5)]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TestTorchBackend(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.data, self.labels = duplicated_data(rng, 25, 400, 8)
        self.rules = [random_rule(rng, 8, depth=3) for _ in range(60)]
        self.rules += [Literal(value=0), Literal(value=3, negated=True)]

    def backends(self, **options):
        for device in available_devices():
            for evaluation in EVALUATIONS:
                yield TorchBackend(device=device, evaluation=evaluation, **options)
                yield TorchBackend(
                    device=device, evaluation=evaluation, batched=True, **options
                )
            yield TorchBackend(device=device, batched=True, batch_size=7, **options)

    def expected_scores(self, fn, dtype=np.float32):
        scores = [
            fn(self.labels, reference_evaluate(rule, self.data)) for rule in self.rules
        ]
        return np.array(scores).astype(dtype)

    def test_evaluations_match_reference(self):
        for device in available_devices():
            data = TorchDataset.from_numpy(
                Dataset(self.data, self.labels), torch.device(device)
            ).data
            for name, evaluate in EVALUATIONS.items():
                for rule in self.rules:
                    with self.subTest(device=device, evaluation=name):
                        np.testing.assert_array_equal(
                            evaluate(rule, data).cpu().numpy(),
                            reference_evaluate(rule, self.data),
                        )

    def test_unsupported_rule_type(self):
        class Xor(Rule):
            pass

        data = torch.ones((2, 2), dtype=torch.bool)
        for evaluate in EVALUATIONS.values():
            with self.assertRaises(TypeError):
                evaluate(Xor([Literal(value=0), Literal(value=1)]), data)

    def test_scores_equal_numpy_backend(self):
        dataset = Dataset(self.data, self.labels)
        for fn in (fast_f1_score, fast_accuracy_score):
            expected = self.expected_scores(fn)
            for merge_rows in (True, False):
                numpy_scores = (
                    NumpyBackend().bind(dataset, fn, merge_rows=merge_rows)
                ).score(self.rules)
                np.testing.assert_array_equal(numpy_scores, expected)
                for backend in self.backends():
                    with self.subTest(
                        fn=fn.__name__, merge=merge_rows, backend=backend
                    ):
                        evaluator = backend.bind(dataset, fn, merge_rows=merge_rows)
                        self.assertIsInstance(evaluator.dataset, TorchDataset)
                        scores = evaluator.score(self.rules)
                        self.assertIsInstance(scores, torch.Tensor)
                        self.assertEqual(scores.device.type, backend.device.type)
                        self.assertEqual(scores.dtype, torch.float32)
                        np.testing.assert_array_equal(
                            backend.to_numpy(scores), expected
                        )
                        self.assertEqual(tuple(evaluator.score([]).shape), (0,))

    def test_float64_scores(self):
        dataset = Dataset(self.data, self.labels)
        expected = self.expected_scores(fast_f1_score, np.float64)
        for merge_rows in (True, False):
            backend = TorchBackend(dtype="float64")
            evaluator = backend.bind(dataset, merge_rows=merge_rows)
            np.testing.assert_array_equal(
                backend.to_numpy(evaluator.score(self.rules)), expected
            )

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

    def test_confusion_matrix(self):
        for backend in self.backends():
            for merge_rows in (True, False):
                evaluator = backend.bind(
                    Dataset(self.data, self.labels), merge_rows=merge_rows
                )
                for rule in self.rules:
                    self.assertEqual(
                        evaluator.confusion_matrix(rule),
                        confusion_matrix(
                            self.labels, reference_evaluate(rule, self.data)
                        ),
                    )

    def test_custom_scorers_receive_tensors(self):
        seen = []

        def spy(y_true, y_pred, sample_weight=None):
            seen.append((y_true, y_pred, sample_weight))
            return custom_weighted_f1(y_true, y_pred, sample_weight)

        expected = self.expected_scores(fast_f1_score)
        for backend in self.backends():
            for merge_rows in (True, False):
                seen.clear()
                evaluator = backend.bind(
                    Dataset(self.data, self.labels), spy, merge_rows=merge_rows
                )
                scores = backend.to_numpy(evaluator.score(self.rules))
                np.testing.assert_allclose(scores, expected, rtol=1e-6)
                for y_true, y_pred, weights in seen:
                    self.assertIsInstance(y_true, torch.Tensor)
                    self.assertIsInstance(y_pred, torch.Tensor)
                    self.assertEqual(y_pred.device.type, backend.device.type)
                    self.assertEqual(weights is not None, merge_rows)

    def test_custom_scorer_with_a_batched_method(self):
        expected = self.expected_scores(fast_f1_score)
        for merge_rows in (True, False):
            scorer = BatchedF1()
            backend = TorchBackend(batched=True, batch_size=7)
            evaluator = backend.bind(
                Dataset(self.data, self.labels), scorer, merge_rows=merge_rows
            )
            scores = backend.to_numpy(evaluator.score(self.rules))
            np.testing.assert_allclose(scores, expected, rtol=1e-6)
            self.assertEqual((scorer.calls, scorer.batched_calls), (0, 9))

    def test_dataset_operations_match_numpy(self):
        rng = np.random.default_rng(8)
        numpy_dataset = Dataset(self.data, self.labels).deduplicate()
        for device in available_devices():
            dataset = TorchDataset.from_numpy(numpy_dataset, torch.device(device))
            self.assertEqual(dataset.n_rows, numpy_dataset.n_rows)
            self.assertEqual(dataset.n_features, numpy_dataset.n_features)
            rows = np.sort(rng.choice(numpy_dataset.n_rows, 150, replace=False))
            columns = np.array([5, 1, 2])
            pairs = [
                (dataset.take(rows), numpy_dataset.take(rows)),
                (
                    dataset.select_features(columns),
                    numpy_dataset.select_features(columns),
                ),
                (
                    dataset.take(rows).select_features(columns).deduplicate(),
                    numpy_dataset.take(rows).select_features(columns).deduplicate(),
                ),
            ]
            for torch_result, numpy_result in pairs:
                converted = torch_result.to_numpy()
                np.testing.assert_array_equal(converted.data, numpy_result.data)
                np.testing.assert_array_equal(converted.labels, numpy_result.labels)
                if numpy_result.sample_weight is None:
                    self.assertIsNone(converted.sample_weight)
                else:
                    np.testing.assert_array_equal(
                        converted.sample_weight, numpy_result.sample_weight
                    )
                # Data stays column-major, so literals read contiguous columns.
                self.assertEqual(torch_result.data.stride(0), 1)

    def test_rows_are_not_copied_on_the_cpu(self):
        data = np.asfortranarray(self.data)
        evaluator = TorchBackend().bind(Dataset(data, self.labels))
        self.assertEqual(evaluator.dataset.data.data_ptr(), data.ctypes.data)

    def test_subset(self):
        rng = np.random.default_rng(9)
        for backend in self.backends():
            for merge_rows in (True, False):
                evaluator = backend.bind(
                    Dataset(self.data, self.labels), merge_rows=merge_rows
                )
                rows = rng.choice(evaluator.dataset.n_rows, 100, replace=False)
                subset = evaluator.subset(rows)
                self.assertIsInstance(subset.dataset, TorchDataset)
                numpy_subset = (
                    NumpyBackend()
                    .bind(Dataset(self.data, self.labels), merge_rows=merge_rows)
                    .subset(rows)
                )
                np.testing.assert_array_equal(
                    backend.to_numpy(subset.score(self.rules)),
                    numpy_subset.score(self.rules),
                )

    def test_predict(self):
        read_only = self.data.copy()
        read_only.setflags(write=False)
        for backend in self.backends():
            for rule in self.rules[:20]:
                expected = reference_evaluate(rule, self.data)
                native = backend.predict(rule, backend.from_numpy(read_only))
                self.assertIsInstance(native, torch.Tensor)
                np.testing.assert_array_equal(backend.to_numpy(native), expected)
                result = predict(rule, read_only, backend=backend)
                self.assertIsInstance(result, np.ndarray)
                np.testing.assert_array_equal(result, expected)

    def test_score_function(self):
        rule = Literal(value=0)
        expected = score(rule, self.data, self.labels)
        for backend in self.backends():
            self.assertEqual(
                score(rule, self.data, self.labels, backend=backend), expected
            )

    def test_read_only_inputs_do_not_warn(self):
        data = self.data.copy()
        data.setflags(write=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TorchBackend().bind(Dataset(data, self.labels))
            TorchBackend().predict(self.rules[0], TorchBackend().from_numpy(data))

    def test_device(self):
        self.assertEqual(TorchBackend().device, torch.device("cpu"))
        self.assertEqual(TorchBackend(), TorchBackend(device="cpu"))
        self.assertEqual(TorchBackend(device=torch.device("cpu")).device.type, "cpu")
        evaluator = TorchBackend().bind(Dataset(self.data, self.labels))
        self.assertEqual(evaluator.dataset.data.device.type, "cpu")
        with self.assertRaises(TypeError):
            TorchBackend(device=0.5)
        with self.assertRaises(RuntimeError):
            TorchBackend(device="not-a-device")

    def test_options_are_validated(self):
        with self.assertRaises(ValueError):
            TorchBackend(evaluation="fast")
        with self.assertRaises(TypeError):
            TorchBackend(batched="yes")
        with self.assertRaises(TypeError):
            TorchBackend(batched=True, batch_size=2.0)
        with self.assertRaises(ValueError):
            TorchBackend(batched=True, batch_size=0)
        with self.assertRaises(ValueError):
            TorchBackend(batch_size=8)
        with self.assertRaises(ValueError):
            TorchBackend(dtype=torch.int32)
        with self.assertRaises(ValueError):
            TorchBackend(dtype="not_a_dtype")
        self.assertEqual(TorchBackend(dtype="float64").dtype, torch.float64)

    @unittest.skipUnless(
        torch is not None and torch.backends.mps.is_available(), "needs MPS"
    )
    def test_mps_has_no_float64(self):
        with self.assertRaises(ValueError):
            TorchBackend(device="mps", dtype=torch.float64)
        with self.assertRaises(ValueError):
            count_dtype(torch.float32, 2**24, torch.device("mps"))

    def test_counts_switch_to_float64_when_float32_is_not_exact(self):
        cpu = torch.device("cpu")
        self.assertEqual(count_dtype(torch.float32, 2**23, cpu), torch.float32)
        self.assertEqual(count_dtype(torch.float32, 2**23 + 1, cpu), torch.float64)
        data = np.array([[True], [True], [False]])
        labels = np.array([True, True, False])
        weights = np.array([2**24, 1, 1])
        evaluator = TorchBackend()._bind(
            Dataset(data, labels, weights), fast_f1_score, True
        )
        self.assertEqual(
            evaluator.confusion_matrix(Literal(value=0)), (2**24 + 1, 0, 0, 1)
        )
        self.assertEqual(evaluator.score([Literal(value=0)]).tolist(), [1.0])

    def test_pickle(self):
        backend = TorchBackend(evaluation="block", batched=True, batch_size=8)
        self.assertEqual(pickle.loads(pickle.dumps(backend)), backend)
        evaluator = backend.bind(Dataset(self.data, self.labels))
        restored = pickle.loads(pickle.dumps(evaluator))
        np.testing.assert_array_equal(
            restored.score(self.rules).numpy(), evaluator.score(self.rules).numpy()
        )

    def test_lazy_export(self):
        from hgp_lib.evaluation import TorchBackend as exported

        self.assertIs(exported, TorchBackend)

    def train(self, backend, **options):
        np.random.seed(0)
        random.seed(0)
        options = {
            "population_factory": PopulationGeneratorFactory(population_size=30),
            **options,
        }
        gp = BooleanGP(
            BooleanGPConfig(
                train_data=self.data,
                train_labels=self.labels,
                backend=backend,
                **options,
            )
        )
        history = [gp.step().best_train_score for _ in range(15)]
        return history, str(gp.global_best_rule)

    def test_training_matches_numpy_backend(self):
        # Same seed and the same float32 scores, so the same rules evolve.
        hierarchical = {
            "max_depth": 1,
            "num_child_populations": 2,
            "sampling_strategy": CombinedSamplingStrategy(
                feature_fraction=0.75, sample_fraction=0.5
            ),
            "top_k_transfer": 5,
        }
        for options in (
            {},
            hierarchical,
            {"population_factory": LiteralPairs(population_size=20)},
        ):
            expected = self.train(NumpyBackend(), **options)
            for backend in self.backends():
                with self.subTest(backend=backend, options=sorted(options)):
                    self.assertEqual(self.train(backend, **options), expected)


if __name__ == "__main__":
    unittest.main()
