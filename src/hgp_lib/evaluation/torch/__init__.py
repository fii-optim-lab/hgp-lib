"""
The PyTorch evaluation backend. Needs PyTorch: ``pip install "hgp-lib[torch]"``.
"""

try:
    import torch  # noqa: F401
except ImportError as error:
    raise ImportError(
        'TorchBackend needs PyTorch. Install it with pip install "hgp-lib[torch]".'
    ) from error

from .backend import TorchBackend, TorchEvaluator

__all__ = ["TorchBackend", "TorchEvaluator"]
