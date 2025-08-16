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

# TODO: Document how to pass the population initializer to the orchestrator.
#  * The plan is like this:
#    - either pass a string literal denoting one of the already implemented initializers
#    - or pass a type, inheriting from PopulationInitializer and implementing generate
#       * if passing a type, the user is also required to pass initializer_kwargs, which will be used to initialize the
#       population; we will also try to do internally introspection to match missing kwarg-values for kwargs already
#       used in our implemented population initializers; e.g. if the user uses `data` for __init__, but doesn't have a
#       `data` key in the initializer_kwargs, we will provide the `data`
