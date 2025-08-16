import random
from typing import List, Type

import numpy as np
from numpy import ndarray

from hgp_lib.populations.initializers.population_initializer import PopulationInitializer
from hgp_lib.rules import Rule, Literal


class RandomInitializer(PopulationInitializer):
    def __init__(self, pop_size: int, data: ndarray, labels: ndarray, available_operators: List[Type[Rule]],
                 max_num_literals: int):
        super().__init__(pop_size, data, labels, available_operators)

        assert isinstance(max_num_literals, int) and max_num_literals > 2, \
            f"The maximum number of literals for random population initialization must be an integer >= 2, " \
            f"is {max_num_literals}"
        # TODO: Add assert for the number of columns in the data, here we need at least 2

        # TODO: Add warning if number of columns is less than max_num_literals. Catch warning in tests
        self.max_num_literals = min(max_num_literals, len(self.columns))

    def generate(self) -> List[Rule]:
        num_chosen_literals = np.random.randint(2, self.max_num_literals, self.pop_size)
        chosen_operators, negated_operators = self.select_operators()
        return [self.generate_one(num_literals, operator, negated_operator) for num_literals, operator, negated_operator
                in zip(num_chosen_literals, chosen_operators, negated_operators)]

    def generate_one(self, num_literals: int, operator: Type[Rule], negated_operator: bool) -> Rule:
        chosen_columns = random.sample(self.columns, num_literals)
        negated_literals = np.random.rand(num_literals) < 0.5
        return operator(
            subrules=[Literal(value=x, negated=n) for x, n in zip(chosen_columns, negated_literals)],
            negated=negated_operator
        )
