import os

from .literals import Literal
from .rules import Rule
from . import utils

if os.getenv("HPG_LOW_MEMORY", "0") == "1":
    from .low_memory_operators import Or, And
else:
    from .operators import Or, And

__all__ = ["Literal", "Rule", "Or", "And", "utils"]
