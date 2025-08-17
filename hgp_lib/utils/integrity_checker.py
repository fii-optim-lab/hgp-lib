from abc import abstractmethod, ABC
from typing import Callable, Optional

from numpy import ndarray

from hgp_lib.rules import Rule


class IntegrityChecker(ABC):
    def __init__(self, mutation_checks: int, crossover_checks: int):
        assert isinstance(mutation_checks, int) and mutation_checks >= 0, \
            f"The number of mutation checks must be a positive integer or 0, is {mutation_checks}"
        assert isinstance(crossover_checks, int) and crossover_checks >= 0, \
            f"The number of mutation checks must be a positive integer or 0, is {mutation_checks}"

        self.mutation_checks = mutation_checks
        self.crossover_checks = crossover_checks

    @abstractmethod
    def check(self, chromosome: Rule, data: ndarray, labels: ndarray) -> bool:
        pass


class DefaultIntegrityChecker(IntegrityChecker):
    # This is the default integrity checker which verifies that we have not made a mutation that decreases our score
    # to 0
    @abstractmethod
    def check(self, chromosome: Rule, data: ndarray, labels: ndarray) -> bool:
        return (chromosome.evaluate(data) == labels).any()
