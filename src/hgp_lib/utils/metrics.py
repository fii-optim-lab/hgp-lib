"""
Deprecated: moved to `hgp_lib.evaluation`, and removed in a future 2.x release.

Importing a scoring function from here still works, with a ``DeprecationWarning``.
The helpers that bound sample weights to scorers were removed in 2.0.0; rows are now
merged by `hgp_lib.evaluation.EvaluationBackend.bind`. See MIGRATION.md.
"""

from .. import evaluation as _evaluation
from .warnings import warn_moved

_MOVED = {
    "accepts_sample_weight",
    "confusion_matrix",
    "fast_accuracy_score",
    "fast_f1_score",
}

_REMOVED = {
    "SampleWeightScorer",
    "optimize_scorers_for_data",
    "select_weighted_rows",
    "transform_duplicates_to_sample_weight",
}


def __getattr__(name: str):
    if name in _MOVED:
        warn_moved(f"hgp_lib.utils.metrics.{name}", f"hgp_lib.evaluation.{name}")
        return getattr(_evaluation, name)
    if name in _REMOVED:
        raise AttributeError(
            f"hgp_lib.utils.metrics.{name} was removed in 2.0.0. Duplicate rows are now "
            "merged by EvaluationBackend.bind (see Dataset.deduplicate and "
            "Dataset.take); see MIGRATION.md."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
