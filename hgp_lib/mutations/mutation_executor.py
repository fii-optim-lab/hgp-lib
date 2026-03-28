import random
from typing import Callable, Tuple, List

import numpy as np

from .base_mutation import Mutation
from .utils import MutationError
from ..rules import Rule, Literal
from ..rules.utils import select_crossover_point


class MutationExecutor:
    """
    Coordinates literal and operator mutations across a collection of `Rule` trees.

    The executor samples how many nodes of every rule should be mutated based on `mutation_p`,
    picks concrete nodes uniformly at random, and retries failed mutations up to `num_tries`
    times when a `check_valid` predicate is provided.

    Args:
        literal_mutations (Tuple[Mutation, ...]):
            Mutations that can be applied to literal nodes. The sequence must be non-empty and
            each entry must declare `is_literal_mutation=True`.
        operator_mutations (Tuple[Mutation, ...]):
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
        operator_p (float):
            Probability of selecting an operator node (vs. a literal) when choosing a mutation
            point in the rule tree. Must be in [0.0, 1.0]. Default: `0.9`.

    Examples:
        >>> import random
        >>> import numpy as np
        >>> from hgp_lib.mutations import MutationExecutor, NegateMutation
        >>> from hgp_lib.rules import Literal, And
        >>> random.seed(1); np.random.seed(0)
        >>> executor = MutationExecutor(
        ...     literal_mutations=[NegateMutation()],
        ...     operator_mutations=[NegateMutation()],
        ...     mutation_p=1.0,
        ...     operator_p=0.5,
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
        literal_mutations: Tuple[Mutation, ...],
        operator_mutations: Tuple[Mutation, ...],
        mutation_p: float = 0.1,
        check_valid: Callable[[Rule], bool] | None = None,
        num_tries: int = 1,
        operator_p: float = 0.5,
    ):
        # mutation_p was checked
        # check_valid was checked
        # num_tries was checked
        # operator_p was checked
        # literal_mutations was checked
        # operator_mutations was checked
        self.mutation_p: float = mutation_p
        self.literal_mutations: Tuple[Mutation, ...] = literal_mutations
        self.operator_mutations: Tuple[Mutation, ...] = operator_mutations
        self.check_valid: Callable[[Rule], bool] | None = check_valid
        self.num_tries: int = num_tries
        self.operator_p: float = operator_p

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
            n_mutations = np.random.binomial(len(rule), self.mutation_p)
            if n_mutations != 0:
                rules[i] = self._mutate(rule, n_mutations)

    def _mutate(self, rule: Rule, n_mutations: int) -> Rule:
        new_rule = rule.copy()

        last_mutation = n_mutations - 1
        last_try = self.num_tries - 1

        for mutation_i in range(n_mutations):
            for tries in range(self.num_tries):
                selected = select_crossover_point(new_rule, operator_p=self.operator_p)
                # selected = random.choice(new_rule.flatten())

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
                elif mutation_i != last_mutation or tries != last_try:
                    new_rule = rule.copy()
        return rule
