"""
The PyTorch evaluation backend.
"""

try:
    import torch  # noqa: F401
except ImportError as error:
    raise ImportError(
        "Using the PyTorch backend requires PyTorch 2.0 or later."
    ) from error

from .backend import TorchBackend, TorchEvaluator
from .dataset import TorchDataset

__all__ = ["TorchBackend", "TorchDataset", "TorchEvaluator"]
