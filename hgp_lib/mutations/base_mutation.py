from abc import ABC, abstractmethod

from numpy.random import Generator

from ..rules import Rule
from ..utils.validation import check_isinstance


class Mutation(ABC):
    """
    Base mutation class that performs inplace mutation for a given `Rule`. Operator and literal mutations must inherit
    from this class and correctly set the literal or operator mutation flags.

    Attributes:
        is_literal_mutation (bool): `True` if mutation is a literal mutation. `False`, otherwise.
        is_operator_mutation (bool): `True` if mutation is an operator mutation. `False`, otherwise. At least one of
            `is_literal_mutation` and `is_operator_mutation` must be `True`.

    Examples:
        >>> from numpy.random import default_rng
        >>> from hgp_lib.mutations import Mutation
        >>> from hgp_lib.rules import Literal
        >>> class MakeTrue(Mutation):
        ...     def __init__(self):
        ...         super().__init__(True, True)  # This is a mutation that works for both literals and operators
        ...
        ...     def apply(self, rule: Rule, rng: Generator):
        ...         rule.negated = False
        ...
        >>> rule = Literal(value=0, negated=True)
        >>> rule
        ~0
        >>> mutation = MakeTrue()
        >>> rng = default_rng(42)
        >>> mutation.apply(rule, rng)
        >>> rule
        0
    """

    def __init__(self, is_literal_mutation: bool, is_operator_mutation: bool):
        check_isinstance(is_literal_mutation, bool)
        check_isinstance(is_operator_mutation, bool)
        if not is_literal_mutation and not is_operator_mutation:
            raise ValueError(
                f"A mutation must be at least either a literal mutation, or an operator mutation. "
                f"{type(self)} is neither."
            )

        self.is_literal_mutation: bool = is_literal_mutation
        self.is_operator_mutation: bool = is_operator_mutation

    @abstractmethod
    def apply(self, rule: Rule, rng: Generator):
        """
        Applies inplace mutation to a given subrule.

        Args:
            rule (Rule): Subrule which will undergo mutation.
            rng (Generator): NumPy random Generator for reproducible randomness.

        Raises:
            MutationError: When the mutation violates rule constraints such as eliminating all subrules from a rule,
                leaving an empty operator (e.g. And()).

        Examples:
            >>> from numpy.random import default_rng
            >>> from hgp_lib.mutations import NegateMutation
            >>> from hgp_lib.rules import Literal
            >>> rng = default_rng(42)
            >>> rule = Literal(value=0, negated=True)
            >>> mutation = NegateMutation()
            >>> mutation.apply(rule, rng)
            >>> rule
            0
            >>> mutation.apply(rule, rng)
            >>> rule
            ~0
        """
        pass
