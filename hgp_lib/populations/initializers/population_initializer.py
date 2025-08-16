from abc import ABC, abstractmethod

from numpy import ndarray


class PopulationInitializer(ABC):
    def __init__(self, pop_size: int):
        # TODO: Add messages for asserts
        assert isinstance(pop_size, int)
        assert pop_size > 0

        self.pop_size = pop_size

    @abstractmethod
    def generate(self):
        pass
