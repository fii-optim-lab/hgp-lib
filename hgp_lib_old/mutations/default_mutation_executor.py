import random
from typing import List, Callable, Sequence

from hgp_lib.mutations.mutation_executor import MutationExecutor, MutationError
from hgp_lib.mutations.standard_mutations import delete
from hgp_lib.mutations.utils import mutation_fn_checker
from hgp_lib.rules import Rule
from hgp_lib.rules.operators import is_operator


class DefaultMutationExecutor(MutationExecutor):
    def __init__(
            self,
            max_tries: int,
            operator_mutation_list: Sequence[Callable[[Rule], int]],
            literal_mutation_list: Sequence[Callable[[Rule], int]]
    ):
        assert isinstance(max_tries, int) and max_tries > 0, f"Max tries must be a positive integer, is {max_tries}"
        assert isinstance(operator_mutation_list, Sequence) and len(operator_mutation_list) and all(
            [callable(x) for x in operator_mutation_list]), \
            f"Operator mutation list must be a non empty list of callables, is {operator_mutation_list}"
        assert isinstance(literal_mutation_list, Sequence) and len(literal_mutation_list) and all(
            [callable(x) for x in literal_mutation_list]), \
            f"Literal mutation list must be a non empty list of callables, is {literal_mutation_list}"
        for mutation_fn in operator_mutation_list + literal_mutation_list:
            mutation_fn_checker(mutation_fn)

        self.max_tries = max_tries
        self.operator_mutation_list = operator_mutation_list
        self.literal_mutation_list = literal_mutation_list

    def mutate_gene(self, gene: Rule) -> int:
        mutation_list = self.operator_mutation_list if is_operator(gene) else self.literal_mutation_list
        for _ in range(self.max_tries):
            try:
                return random.choice(mutation_list)(gene)
            except MutationError:  # Don't apply the mutation
                pass

    def apply_shrink_mutation(self, chromosome: Rule, genes: List[Rule], max_complexity: int) -> Rule:
        for _ in range(self.max_tries):
            try:
                deleted = delete(random.choice(genes))
                if len(genes) - deleted <= max_complexity:
                    return chromosome
                genes = chromosome.flatten()
            except MutationError:  # Don't apply the mutation
                pass
        return chromosome  # We didn't manage to delete anything out of it
