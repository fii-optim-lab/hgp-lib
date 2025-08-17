from typing import Callable

from hgp_lib.populations.mutations.mutation_executor import MutationError
from hgp_lib.rules import Rule, Or, Literal


def mutation_fn_checker(mutate_gene: Callable[[Rule], int]):
    rule_1 = Literal(value=0)
    rule_2 = Or(subrules=[Literal(value=1), Literal(value=2)])
    try:
        rez = mutate_gene(rule_1)
        assert isinstance(rez, int), "Mutate fn must return int"
    except MutationError:
        pass
    try:
        rez = mutate_gene(rule_2)
        assert isinstance(rez, int), "Mutate fn must return int"
    except MutationError:
        pass

