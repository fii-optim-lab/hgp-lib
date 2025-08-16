from abc import ABC, abstractmethod

from numpy import ndarray

from hgp_lib.populations.initializers.population_initializer import PopulationInitializer
from hgp_lib.rules import Or, And, Rule, Literal


class SubsetInitializer(PopulationInitializer, ABC):
    def __init__(self, pop_size: int, data: ndarray, labels: ndarray, sample_ratio: float, feature_ratio: float):
        super().__init__(pop_size)

        assert 0.0 <= sample_ratio <= 1.0, \
            f"Sample ratio for initializer must be between 0.0 and 1.0, is {sample_ratio}"
        assert 0.0 <= feature_ratio <= 1.0, \
            f"Feature ratio for initializer must be between 0.0 and 1.0, is {feature_ratio}"

        # TODO: Add asserts for data checking the API we use
        self.data = data
        self.labels = labels
        self.sample_ratio = sample_ratio
        self.feature_ratio = feature_ratio

    def generate(self):
        return [self.generate_one() for _ in range(self.pop_size)]

    @abstractmethod
    def generate_one(self):
        pass


class BestLiteralOnSubsetInitializer(SubsetInitializer):
    def generate_one(self):
        raise NotImplementedError("TODO: Implement subset and then best literal")


class BestOperatorOnSubsetInitializer(SubsetInitializer):
    def __init__(self, pop_size: int, data: ndarray, labels: ndarray, sample_ratio: float, feature_ratio: float,
                 operator: Or | And, num_literals: int):
        super().__init__(pop_size, data, labels, sample_ratio, feature_ratio)

        assert isinstance(operator, Rule) and not isinstance(operator, Literal), \
            f"Operator must be Or | And, is {operator}"
        assert isinstance(num_literals, int) and num_literals > 0, \
            f"The number of literals must be a positive integer, is {num_literals}"

        self.operator = operator
        self.num_literals = num_literals

    def generate_one(self):
        raise NotImplementedError("TODO: Implement subset and then search for the best operator with the best literals")


# TODO: Also add a pyomo class for solver based initialization
