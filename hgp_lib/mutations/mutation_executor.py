from abc import ABC, abstractmethod
from typing import List

from hgp_lib.rules import Rule


class MutationExecutor(ABC):
    @abstractmethod
    def mutate_gene(self, gene: Rule) -> int:
        # TODO: Write documentation. Applies mutation to gene.
        #  If applying a destructive mutation that affects the next `x` genes (usually
        #  deletion) return `x` such that those genes will be escaped and no mutation will be applied to them.
        pass

    @abstractmethod
    def apply_shrink_mutation(self, chromosome: Rule, genes: List[Rule], max_complexity: int) -> Rule:
        # TODO: Write documentation. Destructive mutation are used for pruning longer rules until they reach the
        #  allowed maximum complexity.
        pass


class MutationError(Exception):
    # Placeholder class used to signify that a mutation could not be executed because it would violate constraints
    pass
