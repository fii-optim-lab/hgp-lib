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
    Scorer,
    fast_accuracy_score,
    fast_f1_score,
    predict,
    resolve_scorer,
)
from hgp_lib.populations import PopulationGeneratorFactory
from hgp_lib.rules import And, Literal

from test_evaluation import (
    custom_weighted_f1,
    duplicated_data,
    random_rule,
    reference_evaluate,
)

try:
    import torch

    from hgp_lib.evaluation.torch import TorchBackend
    from hgp_lib.evaluation.torch.predict import evaluate
except ImportError:  # pragma: no cover - only without PyTorch
    torch = None


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TestTorchBackend(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.data, self.labels = duplicated_data(rng, 25, 400, 8)
        self.rules = [random_rule(rng, 8, depth=3) for _ in range(60)]
        self.rules += [Literal(value=0), Literal(value=3, negated=True)]

    def backends(self):
        for device in available_devices():
            yield TorchBackend(device=device)
            yield TorchBackend(device=device, batched=True)
            yield TorchBackend(device=device, batched=True, batch_size=7)

    def expected_scores(self, fn):
        return np.array(
            [
                fn(self.labels, reference_evaluate(rule, self.data))
                for rule in self.rules
            ]
        )

    def test_evaluate_matches_reference(self):
        columns = torch.tensor(np.ascontiguousarray(self.data.T))
        for rule in self.rules:
            np.testing.assert_array_equal(
                evaluate(rule, columns).numpy(), reference_evaluate(rule, self.data)
            )

    def test_scores_equal_numpy_backend(self):
        # Exact integer counts, so the scores are bit-for-bit the NumPy ones.
        for fn in (fast_f1_score, fast_accuracy_score):
            expected = self.expected_scores(fn)
            for merge_rows in (True, False):
                dataset = Dataset(self.data, self.labels)
                numpy_scores = (
                    NumpyBackend()
                    .bind(dataset, Scorer(fn, merge_rows))
                    .score(self.rules)
                )
                np.testing.assert_array_equal(numpy_scores, expected)
                for backend in self.backends():
                    with self.subTest(
                        fn=fn.__name__, merge=merge_rows, backend=backend
                    ):
                        evaluator = backend.bind(dataset, Scorer(fn, merge_rows))
                        self.assertEqual(
                            evaluator.dataset.sample_weight is not None, merge_rows
                        )
                        np.testing.assert_array_equal(
                            evaluator.score(self.rules), expected
                        )
                        self.assertEqual(evaluator.score([]).shape, (0,))

    def test_no_positive_labels(self):
        labels = np.zeros(len(self.labels), dtype=int)
        rules = [
            Literal(value=0),
            And([Literal(value=0), Literal(value=0, negated=True)]),
        ]
        for backend in self.backends():
            for merge_rows in (True, False):
                evaluator = backend.bind(
                    Dataset(self.data, labels), Scorer(fast_f1_score, merge_rows)
                )
                self.assertEqual(evaluator.score(rules).tolist(), [0.0, 1.0])

    def test_confusion_matrix(self):
        for backend in self.backends():
            for merge_rows in (True, False):
                evaluator = backend.bind(
                    Dataset(self.data, self.labels), Scorer(fast_f1_score, merge_rows)
                )
                for rule in self.rules:
                    self.assertEqual(
                        evaluator.confusion_matrix(rule),
                        NumpyBackend()
                        .bind(Dataset(self.data, self.labels), resolve_scorer())
                        .confusion_matrix(rule),
                    )

    def test_custom_scorers(self):
        seen = []

        def spy(y_true, y_pred, sample_weight=None):
            self.assertIsInstance(y_pred, np.ndarray)
            seen.append(sample_weight)
            return custom_weighted_f1(y_true, y_pred, sample_weight)

        expected = self.expected_scores(fast_f1_score)
        for backend in self.backends():
            for merge_rows in (True, False):
                seen.clear()
                evaluator = backend.bind(
                    Dataset(self.data, self.labels), Scorer(spy, merge_rows)
                )
                np.testing.assert_array_equal(evaluator.score(self.rules), expected)
                self.assertEqual(all(w is not None for w in seen), merge_rows)

    def test_predict(self):
        read_only = self.data.copy()
        read_only.setflags(write=False)
        for device in available_devices():
            backend = TorchBackend(device=device)
            for rule in self.rules:
                expected = reference_evaluate(rule, self.data)
                result = predict(rule, read_only, backend=backend)
                self.assertIsInstance(result, np.ndarray)
                np.testing.assert_array_equal(result, expected)

    def test_read_only_inputs_do_not_warn(self):
        data = self.data.copy()
        data.setflags(write=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TorchBackend().bind(Dataset(data, self.labels), resolve_scorer())
            TorchBackend().predict(self.rules[0], data)

    def test_device(self):
        self.assertEqual(TorchBackend().device, torch.device("cpu"))
        self.assertEqual(TorchBackend(), TorchBackend(device="cpu"))
        self.assertEqual(TorchBackend(device=torch.device("cpu")).device.type, "cpu")
        evaluator = TorchBackend().bind(
            Dataset(self.data, self.labels), resolve_scorer()
        )
        self.assertEqual(evaluator._columns.device.type, "cpu")
        with self.assertRaises(TypeError):
            TorchBackend(device=0.5)
        with self.assertRaises(RuntimeError):
            TorchBackend(device="not-a-device")

    def test_options_are_validated(self):
        with self.assertRaises(TypeError):
            TorchBackend(batched="yes")
        with self.assertRaises(TypeError):
            TorchBackend(batched=True, batch_size=2.0)
        with self.assertRaises(ValueError):
            TorchBackend(batched=True, batch_size=0)
        with self.assertRaises(ValueError):
            TorchBackend(batch_size=8)

    def test_pickle(self):
        backend = TorchBackend(batched=True, batch_size=8)
        self.assertEqual(pickle.loads(pickle.dumps(backend)), backend)
        evaluator = backend.bind(Dataset(self.data, self.labels), resolve_scorer())
        restored = pickle.loads(pickle.dumps(evaluator))
        np.testing.assert_array_equal(
            restored.score(self.rules), evaluator.score(self.rules)
        )

    def test_lazy_export(self):
        from hgp_lib.evaluation import TorchBackend as exported

        self.assertIs(exported, TorchBackend)

    def test_training_matches_numpy_backend(self):
        # Same seed and same exact scores, so the same rules evolve.
        results = []
        for backend in (NumpyBackend(), TorchBackend(), TorchBackend(batched=True)):
            np.random.seed(0)
            random.seed(0)
            gp = BooleanGP(
                BooleanGPConfig(
                    train_data=self.data,
                    train_labels=self.labels,
                    population_factory=PopulationGeneratorFactory(population_size=30),
                    backend=backend,
                )
            )
            for _ in range(15):
                gp.step()
            results.append((gp.global_best_score, str(gp.global_best_rule)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])


if __name__ == "__main__":
    unittest.main()
