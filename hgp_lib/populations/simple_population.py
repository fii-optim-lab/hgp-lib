from typing import List, Optional

import numpy as np

from hgp_lib.populations.initializers.population_initializer import PopulationInitializer
from hgp_lib.rules import Rule


class SimplePopulation:
    def __init__(
            self,
            pop_size: int,
            population_initializer: PopulationInitializer,
            max_complexity: Optional[int],

            mutation_p: float,
    ):
        assert isinstance(pop_size, int) and pop_size > 0, f"Population size must be a positive integer, is {pop_size}"
        assert isinstance(population_initializer, PopulationInitializer)
        assert isinstance(mutation_p, float) and 0 <= mutation_p <= 0, \
            f"Mutation probability must be between 0 and 1, is {mutation_p}"
        assert max_complexity is None or isinstance(max_complexity, int) and max_complexity >= 2, \
            f"Maximum complexity must either be None, or an integer greater or equal to 2, is {max_complexity}"

        self.pop_size = pop_size
        self.population_initializer = population_initializer
        self.mutation_p = mutation_p
        self.max_complexity: int = 9223372036854775807 if max_complexity is None else max_complexity

        self.population: List[Rule] = population_initializer.generate()

    def mutation(self):
        for i in range(self.pop_size):
            chromosome_nodes = self.population[i].flatten()
            if len(chromosome_nodes) > self.max_complexity:
                pass # TODO: Destructive mutations
            else:
                last_index = 0
                for index in np.nonzero(np.random.rand(len(chromosome_nodes)) < self.mutation_p)[0]:
                    if last_index > index:
                        continue  # Skipping already deleted nodes
                    # TODO: Now implement the mutation



