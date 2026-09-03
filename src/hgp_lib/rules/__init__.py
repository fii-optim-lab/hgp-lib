import os

from . import utils
from .literals import Literal
from .rules import Rule

if os.getenv("HGP_LOW_MEMORY", "0") == "1":
    from .low_memory_operators import And, Or
else:
    from .operators import And, Or

__all__ = ["And", "Literal", "Or", "Rule", "low_memory_operators", "operators", "utils"]
# TODO: Provide support for both torch and numpy
