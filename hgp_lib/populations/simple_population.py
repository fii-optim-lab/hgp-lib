from typing import List, Optional

import numpy as np

from hgp_lib.populations.initializers.population_initializer import PopulationInitializer
from hgp_lib.populations.mutations.mutation_executor import MutationExecutor
from hgp_lib.populations.utils.integrity_checker import IntegrityChecker
from hgp_lib.rules import Rule


class SimplePopulation:
    def __init__(
            self,
            pop_size: int,
            population_initializer: PopulationInitializer,

            max_complexity: Optional[int],
            mutation_p: float,
            mutation_executor: MutationExecutor,

            integrity_checker: IntegrityChecker,
    ):
        assert isinstance(pop_size, int) and pop_size > 0, f"Population size must be a positive integer, is {pop_size}"
        assert isinstance(population_initializer, PopulationInitializer)

        assert max_complexity is None or isinstance(max_complexity, int) and max_complexity >= 2, \
            f"Maximum complexity must either be None, or an integer greater or equal to 2, is {max_complexity}"
        assert isinstance(mutation_p, float) and 0 <= mutation_p <= 0, \
            f"Mutation probability must be between 0 and 1, is {mutation_p}"
        assert isinstance(mutation_executor, MutationExecutor)
        assert isinstance(integrity_checker, IntegrityChecker)

        self.pop_size = pop_size
        self.population_initializer = population_initializer
        self.mutation_p = mutation_p
        self.max_complexity: int = 9223372036854775807 if max_complexity is None else max_complexity
        self.mutation_executor = mutation_executor
        self.integrity_checker = integrity_checker

        self.population: List[Rule] = population_initializer.generate()

    def try_mutate(self, genes: List[Rule]):
        last_index = 0
        for index in np.nonzero(np.random.rand(len(genes)) < self.mutation_p)[0]:
            if last_index > index:
                continue  # Skipping already deleted nodes
            last_index = index + self.mutation_executor.mutate_gene(genes[index])

    def mutation(self):
        # TODO: Replace list type hints with Sequence, and try to use tuples instead
        for i in range(self.pop_size):
            genes = self.population[i].flatten()
            if len(genes) > self.max_complexity:
                self.population[i] = self.mutation_executor.apply_shrink_mutation(
                    self.population[i], genes, self.max_complexity)
            else:
                if self.integrity_checker.mutation_checks < 2:  # Fast path when no checks are done
                    self.try_mutate(genes)
                else:
                    for _ in range(self.integrity_checker.mutation_checks):
                        backup = self.population[i].copy()
                        self.try_mutate(genes)
                        if self.integrity_checker.check(self.population[i], self.data, self.labels):
                            break  # Early stopping, mutation was ok
                        self.population[i] = backup
                        genes = self.population[i].flatten()
