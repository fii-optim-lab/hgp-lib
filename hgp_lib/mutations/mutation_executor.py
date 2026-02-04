import random
from typing import Callable, Sequence, Tuple, List

import numpy as np

from .base_mutation import Mutation
from .utils import MutationError
from ..rules import Rule, Literal
from ..utils.validation import validate_callable, check_isinstance


class MutationExecutor:
    """
    Coordinates literal and operator mutations across a collection of `Rule` trees.

    The executor samples how many nodes of every rule should be mutated based on `mutation_p`,
    picks concrete nodes uniformly at random, and retries failed mutations up to `num_tries`
    times when a `check_valid` predicate is provided.

    Args:
        literal_mutations (Sequence[Mutation]):
            Mutations that can be applied to literal nodes. The sequence must be non-empty and
            each entry must declare `is_literal_mutation=True`.
        operator_mutations (Sequence[Mutation]):
            Mutations that can be applied to operator nodes. The sequence must be non-empty and
            each entry must declare `is_operator_mutation=True`.
        mutation_p (float):
            Probability of mutating each node inside a rule. Default: `0.1`.
        check_valid (Callable[[Rule], bool] | None):
            Optional validator executed after every successful mutation. When supplied, the
            mutated rule is only kept if the predicate returns `True`. Default: `None`.
        num_tries (int):
            Maximum number of attempts per node in case mutations raise `MutationError` or fail
            validation. Must be `1` when no validator is provided. Default: `1`.

    Examples:
        >>> import random
        >>> import numpy as np
        >>> from hgp_lib.mutations import MutationExecutor, NegateMutation
        >>> from hgp_lib.rules import Literal, And
        >>> random.seed(0); np.random.seed(0)
        >>> executor = MutationExecutor(
        ...     literal_mutations=[NegateMutation()],
        ...     operator_mutations=[NegateMutation()],
        ...     mutation_p=1.0,
        ... )
        >>> rules = [Literal(value=0), And([Literal(value=0), Literal(value=1)])]
        >>> executor.apply(rules)
        >>> str(rules[0])
        '~0'
        >>> str(rules[1])
        '~And(~0, ~1)'
    """

    def __init__(
        self,
        literal_mutations: Sequence[Mutation],
        operator_mutations: Sequence[Mutation],
        mutation_p: float = 0.1,
        check_valid: Callable[[Rule], bool] | None = None,
        num_tries: int = 1,
    ):
        self._validate_params(
            mutation_p, literal_mutations, operator_mutations, check_valid, num_tries
        )
        self.mutation_p: float = mutation_p
        self.literal_mutations: Tuple = tuple(literal_mutations)
        self.operator_mutations: Tuple = tuple(operator_mutations)
        self.check_valid: Callable[[Rule], bool] | None = check_valid
        self.num_tries: int = num_tries

    @staticmethod
    def _validate_params(
        mutation_p: float,
        literal_mutations: Sequence[Mutation],
        operator_mutations: Sequence[Mutation],
        check_valid: Callable[[Rule], bool] | None,
        num_tries: int,
    ):
        check_isinstance(mutation_p, float)
        check_isinstance(literal_mutations, Sequence)
        check_isinstance(operator_mutations, Sequence)
        check_isinstance(num_tries, int)

        if mutation_p < 0.0 or mutation_p > 1.0:
            raise ValueError(
                f"mutation_p must be a float between 0.0 and 1.0, is '{mutation_p}'"
            )

        if len(literal_mutations) == 0:
            raise ValueError("literal_mutations must be a non-empty Sequence")
        if len(operator_mutations) == 0:
            raise ValueError("operator_mutations must be a non-empty Sequence")

        for literal_mutation in literal_mutations:
            check_isinstance(literal_mutation, Mutation)
            if not literal_mutation.is_literal_mutation:
                raise TypeError(
                    f"Each literal_mutations must be a literal mutation, but '{type(literal_mutation)} is not'"
                )
        for operator_mutation in operator_mutations:
            check_isinstance(operator_mutation, Mutation)
            if not operator_mutation.is_operator_mutation:
                raise TypeError(
                    f"Each operator_mutations must be an operator mutation, but '{type(operator_mutation)} is not'"
                )

        if check_valid is not None:
            error_msg = f"check_valid must be a callable that accepts a Rule and returns bool, is {type(check_valid)}"
            validate_callable(check_valid, error_msg)
            try:
                boolean = check_valid(Literal(value=0))
                if not isinstance(boolean, bool):
                    raise TypeError(error_msg)
            except Exception as e:
                raise TypeError(error_msg) from e

        if num_tries < 1:
            raise ValueError(f"num_tries must be greater than 0, is '{num_tries}'")
        if num_tries > 1 and check_valid is None:
            raise ValueError("num_tries must be 1 if check_valid is None")

    def apply(self, rules: List[Rule]):
        """
        Mutates the provided list of rules in place.

        In the event of a mutation failure, the original rule is kept in place.

        Args:
            rules (List[Rule]):
                The mutable collection of rules that will be potentially replaced by mutated
                versions depending on `mutation_p`.

        Examples:
            >>> import random
            >>> import numpy as np
            >>> from hgp_lib.mutations import MutationExecutor, NegateMutation
            >>> from hgp_lib.rules import Literal
            >>> random.seed(1); np.random.seed(1)
            >>> executor = MutationExecutor(
            ...     literal_mutations=[NegateMutation()],
            ...     operator_mutations=[NegateMutation()],
            ...     mutation_p=1.0,
            ... )
            >>> rules = [Literal(value=0)]
            >>> executor.apply(rules)
            >>> str(rules[0])
            '~0'
        """
        for i in range(len(rules)):
            rule = rules[i]
            # Check this!
            n_mutations = (np.random.rand(len(rule)) < self.mutation_p).sum()
            if n_mutations != 0:
                rules[i] = self._mutate(rule, n_mutations)

    def _mutate(self, rule: Rule, n_mutations: int) -> Rule:
        for _ in range(n_mutations):
            new_rule = rule.copy()
            # TODO: Check random choice of rule without flattening
            flattened = new_rule.flatten()
            for _ in range(self.num_tries):
                selected = random.choice(flattened)

                try:
                    random.choice(
                        self.literal_mutations
                        if isinstance(selected, Literal)
                        else self.operator_mutations
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
