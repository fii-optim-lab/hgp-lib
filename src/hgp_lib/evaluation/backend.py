from abc import ABC, abstractmethod

import numpy as np


class EvaluationBackend(ABC):
    @abstractmethod
    def prepare(self, data: np.ndarray, labels: np.ndarray, score_fn):
        pass