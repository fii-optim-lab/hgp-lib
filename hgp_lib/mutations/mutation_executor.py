import random
from typing import Callable, Sequence, Tuple, List

import numpy as np

from ..rules import Rule, Literal


class MutationExecutor:
    def __init__(self, mutation_p: float = 0.1, mutations: Sequence = None, check_valid: Callable[[Rule], bool] | None = None,
                 num_tries: int = 1):
        self._validate_params(mutation_p, mutations, check_valid, num_tries)
        self.mutation_p: float = mutation_p
        self.mutations: Tuple = tuple(mutations)
        self.check_valid: Callable[[Rule], bool] | None = check_valid
        self.num_tries: int = num_tries

    @staticmethod
    def _validate_params(mutation_p: float = 0.1, mutations: Sequence = None,
                         check_valid: Callable[[Rule], bool] | None = None,
                         num_tries: int = 1):
        """Validate initialization parameters."""
        if not isinstance(mutation_p, float):
            raise TypeError(f"mutation_p must be a float, is '{type(mutation_p)}'")
        if mutation_p < 0.0 or mutation_p > 1.0:
            raise ValueError(f"mutation_p must be a float between 0.0 and 1.0, is '{mutation_p}'}")

        if not isinstance(mutations, Sequence):
            raise TypeError(f"mutations must be a Sequence, is '{type(mutations)}'")
        if len(mutations) == 0:
            raise ValueError(f"mutations must be a non-empty Sequence")
        for mutation in mutations:
            if not isinstance(mutation, str):  # TODO: Change this
                raise TypeError(f"Each element in mutations must be TODO, but one element is '{type(mutation)}'")


        if check_valid is not None:
            error_msg = f"check_valid must be a callable that accepts a Rule and returns bool, is {type(check_valid)}"
            if not callable(check_valid):
                raise TypeError(error_msg)
            try:
                boolean = check_valid(Literal(value=0))
                if not isinstance(boolean, bool):
                    raise TypeError(error_msg)
            except:
                raise TypeError(error_msg)

        if not isinstance(num_tries, int):
            raise TypeError(f"num_tries must be an int, is '{type(num_tries)}'")
        if num_tries < 1:
            raise ValueError(f"num_tries must be greater than 0, is '{num_tries}'")
        if num_tries > 1 and check_valid is None:
            raise ValueError(f"num_tries must be 1 if check_valid is None")


    def apply(self, rules: List[Rule]):
        """
        Mutates the list of rules inplace.

        Args:
            rules (List[Rule]): TODO: Add better documentation + doctest
        """
        for i in range(len(rules)):
            rule = rules[i]
            n_mutations = (np.random.rand(len(rule)) < self.mutation_p).sum()
            if n_mutations != 0:
                rules[i] = self._mutate(rule, n_mutations)

    def _mutate(self, rule: Rule, n_mutations: int) -> Rule:
        for _ in range(n_mutations):
            for _ in range(self.num_tries):
                new_rule = random.choice(self.mutations).apply(rule.copy())
                if self.check_valid is None or self.check_valid(new_rule):
                    rule = new_rule
                    break
        return rule
