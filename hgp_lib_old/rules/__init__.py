import os

from hgp_lib.rules.literals import Literal
from hgp_lib.rules.rules import Rule

if os.getenv("HPG_LOW_MEMORY", "0") == "1":
    from hgp_lib.rules.low_memory_operators import Or, And
else:
    from hgp_lib.rules.operators import Or, And
