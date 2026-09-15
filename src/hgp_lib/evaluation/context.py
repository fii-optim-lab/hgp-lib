from dataclasses import dataclass
from typing import Any

from .score_utils import PreparedScorer


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    data: Any
    labels: Any
    scorer: PreparedScorer

# TODO: I need something that is both fast and easy to use (with multiple scorers)
