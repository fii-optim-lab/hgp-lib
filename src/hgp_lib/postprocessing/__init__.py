"""
Postprocess evolved rules: simplify them, merge the literals of one original column,
and convert them to a canonical DNF or CNF.

`postprocess` returns a `Rule` over the same features; `postprocess_to_str` writes the
result over the original columns, where adjacent bins become one interval and a
constant rule is ``True`` or ``False``. `BooleanRuleClassifier.postprocess` and
`BooleanRuleClassifier.postprocess_to_str` do the same for a fitted classifier.
"""

from . import domain, formatting, normal_form, optimize, tree
from .api import CanonicalForm, postprocess, postprocess_to_str

__all__ = [
    "CanonicalForm",
    "domain",
    "formatting",
    "normal_form",
    "optimize",
    "postprocess",
    "postprocess_to_str",
    "tree",
]
