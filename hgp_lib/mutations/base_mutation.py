from abc import ABC, abstractmethod

from ..rules import Rule


class Mutation(ABC):
    """
    Base mutation class that performs inplace mutation for a given `Rule`. Operator and literal mutations must inherit
    from this class and correctly set the literal or operator mutation flags.

    Attributes:
        is_literal_mutation (bool): `True` if mutation is a literal mutation. `False`, otherwise.
        is_operator_mutation (bool): `True` if mutation is an operator mutation. `False`, otherwise. At least one of
            `is_literal_mutation` and `is_operator_mutation` must be `True`.

    Examples:
        >>> from hgp_lib.mutations import Mutation
        >>> from hgp_lib.rules import Literal
        >>> class MakeTrue(Mutation):
        ...     def __init__(self):
        ...         super().__init__(True, True)  # This is a mutation that works for both literals and operators
        ...
        ...     def apply(self, rule: Rule):
        ...         rule.negated = False
        ...
        >>> rule = Literal(value=0, negated=True)
        >>> rule
        ~0
        >>> mutation = MakeTrue()
        >>> mutation.apply(rule)
        >>> rule
        0
    """

    def __init__(self, is_literal_mutation: bool, is_operator_mutation: bool):
        if not isinstance(is_literal_mutation, bool):
            raise TypeError(
                f"is_literal_mutation must be a bool, is '{type(is_literal_mutation)}'"
            )
        if not isinstance(is_operator_mutation, bool):
            raise TypeError(
                f"is_operator_mutation must be a bool, is '{type(is_operator_mutation)}'"
            )
        if not is_literal_mutation and not is_operator_mutation:
            raise ValueError(
                f"A mutation must be at least either a literal mutation, or an operator mutation. "
                f"{type(self)} is neither."
            )

        self.is_literal_mutation: bool = is_literal_mutation
        self.is_operator_mutation: bool = is_operator_mutation

    @abstractmethod
    def apply(self, rule: Rule):
        """
        Applies inplace mutation to a given subrule.

        Args:
            rule (Rule): Subrule which will undergo mutation

        Raises:
            MutationError: When the mutation violates rule constraints such as eliminating all subrules from a rule,
                leaving an empty operator (e.g. And()).

        Examples:
            >>> from hgp_lib.mutations import NegateMutation
            >>> from hgp_lib.rules import Literal
            >>> rule = Literal(value=0, negated=True)
            >>> mutation = NegateMutation()
            >>> mutation.apply(rule)
            >>> rule
            0
            >>> mutation.apply(rule)
            >>> rule
            ~0
        """
        pass
