"""
Deprecated: renamed to `hgp_lib.results`, and removed in a future 2.x release.

Importing a name from here still works, with a ``DeprecationWarning``.
"""

from .. import results as _results
from ..utils.warnings import warn_moved

_MOVED = frozenset(_results.__all__)


def __getattr__(name: str):
    if name in _MOVED:
        warn_moved(f"hgp_lib.metrics.{name}", f"hgp_lib.results.{name}")
        return getattr(_results, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
