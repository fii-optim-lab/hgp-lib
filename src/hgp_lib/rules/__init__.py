import os

from .literals import Literal
from .rules import Rule

if os.getenv("HGP_LOW_MEMORY", "0") == "1":
    from .low_memory_operators import And, Or
else:
    from .operators import And, Or

from . import utils
from .utils import deserialize, serialize

__all__ = [
    "And",
    "Literal",
    "Or",
    "Rule",
    "deserialize",
    "low_memory_operators",
    "operators",
    "serialize",
    "utils",
]
# TODO: Provide support for both torch and numpy
