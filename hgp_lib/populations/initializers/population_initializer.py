import random
from abc import ABC, abstractmethod
from typing import List, Iterable, Type, Tuple

import numpy as np
from numpy import ndarray

from hgp_lib.rules import Rule


class PopulationInitializer(ABC):
    def __init__(self, pop_size: int, data: ndarray, labels: ndarray, available_operators: List[Type[Rule]]):
        # TODO: Add messages for asserts. If cleaner, separate the asserts
        assert isinstance(pop_size, int)
        assert pop_size > 0
        assert isinstance(available_operators, Iterable) and len(available_operators) > 0
        # all([isinstance(x, Rule) and not isinstance(x, Literal) for x in available_operators])
        # TODO: Check how to check if the type is ok

        self.pop_size = pop_size
        self.data = data
        self.labels = labels
        self.available_operators = available_operators
        self.columns = tuple(range(self.data.shape[1]))

    @abstractmethod
    def generate(self) -> List[Rule]:
        pass

    def select_operators(self) -> Tuple[ndarray, List[Type[Rule]]]:
        chosen_operators = random.choices(self.available_operators, k=self.pop_size)
        negated_operators = np.random.rand(self.pop_size) < 0.5
        return chosen_operators, negated_operators

# TODO: Document how to pass the population initializer to the orchestrator.
#  * The plan is like this:
#    - either pass a string literal denoting one of the already implemented initializers
#    - or pass a type, inheriting from PopulationInitializer and implementing generate
#       * if passing a type, the user is also required to pass initializer_kwargs, which will be used to initialize the
#       population; we will also try to do internally introspection to match missing kwarg-values for kwargs already
#       used in our implemented population initializers; e.g. if the user uses `data` for __init__, but doesn't have a
#       `data` key in the initializer_kwargs, we will provide the `data`
