from abc import ABC, abstractmethod
from typing import List, Type

from numpy import ndarray

from hgp_lib.populations.initializers.population_initializer import PopulationInitializer
from hgp_lib.rules import Or, And, Rule, Literal


class SubsetInitializer(PopulationInitializer, ABC):
    def __init__(self, pop_size: int, data: ndarray, labels: ndarray, available_operators: List[Type[Rule]],
                 sample_ratio: float, feature_ratio: float):
        super().__init__(pop_size, data, labels, available_operators)

        assert 0.0 <= sample_ratio <= 1.0, \
            f"Sample ratio for initializer must be between 0.0 and 1.0, is {sample_ratio}"
        assert 0.0 <= feature_ratio <= 1.0, \
            f"Feature ratio for initializer must be between 0.0 and 1.0, is {feature_ratio}"
        # TODO: Add check that check that subset of features still preserves at least 2 columns

        self.sample_ratio = sample_ratio
        self.feature_ratio = feature_ratio

        # TODO: Implement sample methods here, for samples and features

    def generate(self) -> List[Rule]:
        return [self.generate_one() for _ in range(self.pop_size)]

    @abstractmethod
    def generate_one(self) -> Rule:
        pass


class BestLiteralOnSubsetInitializer(SubsetInitializer):
    def generate_one(self) -> Rule:
        raise NotImplementedError("TODO: Implement subset and then best literal")


class BestOperatorOnSubsetInitializer(SubsetInitializer):
    def __init__(self, pop_size: int, data: ndarray, labels: ndarray, available_operators: List[Type[Rule]],
                 sample_ratio: float, feature_ratio: float, max_num_literals: int):
        super().__init__(pop_size, data, labels, available_operators, sample_ratio, feature_ratio)

        assert isinstance(max_num_literals, int) and max_num_literals > 2, \
            f"The maximum number of literals for random population initialization must be an integer >= 2, " \
            f"is {max_num_literals}"
        # TODO: Add assert for the number of columns in the data, here we need at least 2

    def generate_one(self) -> [Rule]:
        raise NotImplementedError("TODO: Implement subset and then search for the best operator with the best literals")

# TODO: Also add a pyomo class for solver based initialization
