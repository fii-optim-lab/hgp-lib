from .base_mutation import Mutation
from .utils import MutationError
from ..rules import Rule


class DeleteMutation(Mutation):
    def __init__(self):
        super().__init__(is_literal_mutation=True, is_operator_mutation=True)

    def apply(self, rule: Rule):
        parent = rule.parent
        # TODO: @? Should we keep 2 as the limit? Maybe we can let operators have 1 subrule?
        if parent is None or len(parent.subrules) == 2:
            raise MutationError()
        for i in range(len(parent.subrules)):
            if parent.subrules[i] is rule:  # We use reference checking because it is faster
                del parent.subrules[i]
                return
        raise RuntimeError("Unreachable code")


class NegateMutation(Mutation):
    def __init__(self):
        super().__init__(is_literal_mutation=True, is_operator_mutation=True)

    def apply(self, rule: Rule):
        rule.negated = not rule.negated



