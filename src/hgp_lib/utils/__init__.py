"""
Internal helpers shared across hgp_lib: argument validation and warnings.

Nothing here is part of the public API. ``hgp_lib.utils.ComplexityCheck`` and
``hgp_lib.utils.metrics`` still work in 2.x as deprecated aliases.
"""

from .warnings import warn_moved

__all__: list[str] = []


def __getattr__(name: str):
    # ComplexityCheck moved to hgp_lib.rules in 2.0.0.
    if name == "ComplexityCheck":
        from ..rules import ComplexityCheck

        warn_moved("hgp_lib.utils.ComplexityCheck", "hgp_lib.rules.ComplexityCheck")
        return ComplexityCheck
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
