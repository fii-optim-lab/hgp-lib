"""
Rule evaluation: predictions, scores and confusion matrices.

- `predict` and `score` evaluate one rule on binarized data, as given.
- `fast_f1_score`, `fast_accuracy_score` and `confusion_matrix` score boolean predictions.
- Backends (`NumpyBackend`, and `TorchBackend` with PyTorch installed) evaluate many
  rules on the same data during training. `EvaluationBackend.bind` binds a `Scorer` to
  a `Dataset` and returns an `Evaluator`.
"""

from .api import predict, score
from .backend import EvaluationBackend, Evaluator
from .dataset import Dataset
from .numpy import NumpyBackend
from .scorers import (
    Scorer,
    accepts_sample_weight,
    confusion_matrix,
    fast_accuracy_score,
    fast_f1_score,
    resolve_scorer,
)

__all__ = [
    "Dataset",
    "EvaluationBackend",
    "Evaluator",
    "NumpyBackend",
    "Scorer",
    "accepts_sample_weight",
    "confusion_matrix",
    "fast_accuracy_score",
    "fast_f1_score",
    "predict",
    "resolve_scorer",
    "score",
]


def __getattr__(name: str):
    # Imported on first use, so hgp_lib does not need PyTorch.
    if name == "TorchBackend":
        from .torch import TorchBackend

        return TorchBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
