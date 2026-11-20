"""Every backend and option combination must pass `hgp_lib.testing.check_backend`."""

import itertools
import unittest

import numpy as np

from hgp_lib.evaluation import NumpyBackend
from hgp_lib.rules import Rule
from hgp_lib.testing import check_backend

try:
    import torch

    from hgp_lib.evaluation.torch import TorchBackend
except ImportError:  # pragma: no cover - only without PyTorch
    torch = None

EVALUATIONS = ("block", "inplace", "fused")


def torch_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


class BrokenBackend(NumpyBackend):
    """Gives wrong predictions for negated operators."""

    def predict(self, rule: Rule, data: np.ndarray) -> np.ndarray:
        result = super().predict(rule, data)
        return ~result if rule.value is None and rule.negated else result


class TestNumpyBackendConformance(unittest.TestCase):
    def test_every_option_combination(self):
        for order, evaluation, batched in itertools.product(
            ("F", "C"), EVALUATIONS, (False, True)
        ):
            backend = NumpyBackend(order=order, evaluation=evaluation, batched=batched)
            with self.subTest(backend=backend):
                check_backend(backend)

    def test_batch_size_and_dtype(self):
        check_backend(NumpyBackend(batched=True, batch_size=7))
        check_backend(NumpyBackend(dtype=np.float64))

    def test_detects_a_broken_backend(self):
        with self.assertRaises(AssertionError) as context:
            check_backend(BrokenBackend())
        self.assertIn("predict", str(context.exception))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TestTorchBackendConformance(unittest.TestCase):
    def test_every_option_combination(self):
        for device, evaluation, batched in itertools.product(
            torch_devices(), EVALUATIONS, (False, True)
        ):
            backend = TorchBackend(
                device=device, evaluation=evaluation, batched=batched
            )
            with self.subTest(backend=backend):
                check_backend(backend)

    def test_batch_size_and_dtype(self):
        check_backend(TorchBackend(batched=True, batch_size=7))
        check_backend(TorchBackend(dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
