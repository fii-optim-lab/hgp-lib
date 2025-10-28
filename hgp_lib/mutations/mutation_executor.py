import random
from typing import Callable, Sequence, Tuple, List

import numpy as np

from .base_mutation import Mutation
from .utils import MutationError
from ..rules import Rule, Literal


# TODO: Add unittests and docstring and doctests for MutationExecutor

class MutationExecutor:
    def __init__(self, literal_mutations: Sequence[Mutation], operator_mutations: Sequence[Mutation],
                 mutation_p: float = 0.1, check_valid: Callable[[Rule], bool] | None = None, num_tries: int = 1):
        self._validate_params(mutation_p, literal_mutations, operator_mutations, check_valid, num_tries)
        self.mutation_p: float = mutation_p
        self.literal_mutations: Tuple = tuple(literal_mutations)
        self.operator_mutations: Tuple = tuple(operator_mutations)
        self.check_valid: Callable[[Rule], bool] | None = check_valid
        self.num_tries: int = num_tries

    @staticmethod
    def _validate_params(mutation_p: float, literal_mutations: Sequence[Mutation],
                         operator_mutations: Sequence[Mutation], check_valid: Callable[[Rule], bool] | None,
                         num_tries: int):
        """Validate initialization parameters."""
        if not isinstance(mutation_p, float):
            raise TypeError(f"mutation_p must be a float, is '{type(mutation_p)}'")
        if mutation_p < 0.0 or mutation_p > 1.0:
            raise ValueError(f"mutation_p must be a float between 0.0 and 1.0, is '{mutation_p}'")

        if not isinstance(literal_mutations, Sequence):
            raise TypeError(f"literal_mutations must be a Sequence, is '{type(literal_mutations)}'")
        if len(literal_mutations) == 0:
            raise ValueError("literal_mutations must be a non-empty Sequence")
        for mutation in literal_mutations:
            if not isinstance(mutation, Mutation):
                raise TypeError(f"Each literal_mutations must be Mutation, but one element is '{type(mutation)}'")
            if not mutation.is_literal_mutation:
                raise TypeError(f"Each literal_mutations must be a literal mutation, but '{type(mutation)} is not'")

        if not isinstance(operator_mutations, Sequence):
            raise TypeError(f"operator_mutations must be a Sequence, is '{type(operator_mutations)}'")
        if len(operator_mutations) == 0:
            raise ValueError("operator_mutations must be a non-empty Sequence")
        for mutation in operator_mutations:
            if not isinstance(mutation, Mutation):
                raise TypeError(f"Each operator_mutations must be Mutation, but one element is '{type(mutation)}'")
            if not mutation.is_operator_mutation:
                raise TypeError(f"Each operator_mutations must be an operator mutation, but '{type(mutation)} is not'")

        if check_valid is not None:
            error_msg = f"check_valid must be a callable that accepts a Rule and returns bool, is {type(check_valid)}"
            if not callable(check_valid):
                raise TypeError(error_msg)
            try:
                boolean = check_valid(Literal(value=0))
                if not isinstance(boolean, bool):
                    raise TypeError(error_msg)
            except Exception as e:
                raise TypeError(error_msg) from e

        if not isinstance(num_tries, int):
            raise TypeError(f"num_tries must be an int, is '{type(num_tries)}'")
        if num_tries < 1:
            raise ValueError(f"num_tries must be greater than 0, is '{num_tries}'")
        if num_tries > 1 and check_valid is None:
            raise ValueError("num_tries must be 1 if check_valid is None")

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
            new_rule = rule.copy()
            flattened = new_rule.flatten()
            for _ in range(self.num_tries):
                selected = random.choice(flattened)

                try:
                    random.choice(
                        self.literal_mutations if isinstance(selected, Literal) else self.operator_mutations
                    ).apply(selected)
                except MutationError:
                    continue

                if self.check_valid is None or self.check_valid(new_rule):
                    rule = new_rule
                    break
                else:
                    new_rule = rule.copy()
                    flattened = new_rule.flatten()
        return rule
