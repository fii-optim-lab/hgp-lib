from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    data: Any
    labels: Any
    score_fn: Callable

# TODO: I need something that is both fast and easy to use (with multiple scorers)
