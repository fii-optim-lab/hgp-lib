from collections.abc import Callable

import numpy as np

from ...rules import Rule
from .. import EvaluationBackend, EvaluationContext
from ..scorer import transform_duplicates_to_sample_weight, optimize_scorers_for_data
from .f1_score import create_fast_np_f1_score


class NumpyBackend(EvaluationBackend):
    def __init__(self):
        # TODO: I should have the following two flags:
        #  * low_memory => default to false, implements the low memory evaluation if True. Docs should state low_memory is usually slower, but we will compare this in the future
        #  * batched => default to false, if true uses the batched version, docs should state that if batched is true, then the scoring fn provided by the user must work with both y_true, y_pred and also y_true, y_preds (for batched eval).
        pass
    def prepare_evaluation_context(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        score_fn: Callable | None,
        optimize_scorer: bool,
    ) -> EvaluationContext:
        # Optimization should always be enabled, maybe we should disable the possibility to not optimize scorer
        if optimize_scorer and score_fn is None:
            # TODO: we should also implement batched version
            data, labels, score_fn = create_fast_np_f1_score(data, labels)
        elif optimize_scorer:
            score_fn, data, labels = optimize_scorers_for_data(score_fn, data=data, labels=labels)
        return EvaluationContext(data, labels, score_fn=score_fn)

    def evaluate_population(
        self, population: list[Rule], evaluation: EvaluationContext
    ) -> np.ndarray:
        pass

    def evaluate_rule(self, rule: Rule, data: np.ndarray) -> np.ndarray:
        pass
