from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    data: Any
    labels: Any
    score_fn: Callable
