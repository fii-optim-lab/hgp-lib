import numpy as np

from .rules import Rule


# TODO: Add some performance tests comparing operator implementations


class And(Rule):
    """
    Logical conjunction (`AND`) operator node for rule trees. It evaluates to `True` only if every subrule evaluates to
    `True`.

    Attributes:
        subrules (List[Rule]):
            A list of child rules combined with logical AND. Must be a list longer than 1 element. Not checked at
            runtime for performance reasons. Default: `None`.
        parent (Optional[Rule]):
            A reference to the parent rule, if part of a larger tree. Default: `None`.
        value (None):
            Always `None` for operator nodes (non-literals). Not checked at runtime for performance reasons. Default:
            `None`.
        negated (bool):
            Whether the entire conjunction is logically negated (`~And(...)`). Default: `False`.

    Examples:
        >>> from hgp_lib.rules.low_memory_operators import And, Or
        >>> from hgp_lib.rules import Literal
        >>> rule = And([
        ...     Literal(value=0),
        ...     Or([Literal(value=1, negated=True), Literal(value=2)]),
        ...     Literal(value=3)
        ... ])
        >>> rule
        And(0, Or(~1, 2), 3)
    """

    def evaluate(self, data: np.ndarray) -> np.ndarray:
        """
        All subrules are recursively evaluated, and their results are combined using logical conjunction. The final
        result is optionally negated if `self.negated` is `True`.

        Args:
            data (np.ndarray):
                Input data passed to subrules. Must be a 2D ndarray, with instances on rows and features on columns.
                Not checked at runtime for performance reasons.

        Returns:
            np.ndarray:
                The boolean result of evaluating this rule vectorized across all instances.

        Examples:
            >>> from hgp_lib.rules.low_memory_operators import And
            >>> from hgp_lib.rules import Literal
            >>> import numpy as np
            >>> data = np.array([
            ...     [True, False],
            ...     [False, False],
            ...     [False, False]
            ... ])
            >>> rule = And([Literal(value=0), Literal(value=1, negated=True)])
            >>> rule.evaluate(data)
            array([ True, False, False])

        Notes:
            This implementation is efficient and allocates as little memory as possible.
        """
        rez = self.subrules[0].evaluate(data)
        if self.subrules[0].value is not None and not self.subrules[0].negated:
            rez = rez.copy()
        for subrule in self.subrules[1:]:
            rez &= subrule.evaluate(data)
        if self.negated:
            rez = np.logical_not(rez, out=rez)
        return rez


class Or(Rule):
    """
    Logical disjunction (`OR`) operator node for rule trees. It evaluates to `True` if any subrule evaluates to `True`.

    Attributes:
        subrules (List[Rule]):
            A list of child rules combined with logical AND. Must be a list longer than 1 element. Not checked at
            runtime for performance reasons. Default: `None`.
        parent (Optional[Rule]):
            A reference to the parent rule, if part of a larger tree. Default: `None`.
        value (None):
            Always `None` for operator nodes (non-literals). Not checked at runtime for performance reasons. Default:
            `None`.
        negated (bool):
            Whether the entire conjunction is logically negated (`~And(...)`). Default: `False`.

    Examples:
        >>> from hgp_lib.rules.low_memory_operators import And
        >>> from hgp_lib.rules import Literal
        >>> rule = Or([Literal(value=0), Literal(value=1, negated=True)])
        >>> rule
        Or(0, ~1)
    """

    def evaluate(self, data: np.ndarray) -> np.ndarray:
        """
        All subrules are recursively evaluated, and their results are combined using logical disjunction. The final
        result is optionally negated if `self.negated` is `True`.

        Args:
            data (np.ndarray):
                Input data passed to subrules. Must be a 2D ndarray, with instances on rows and features on columns.
                Not checked at runtime for performance reasons.

        Returns:
            np.ndarray:
                The boolean result of evaluating this rule vectorized across all instances.

        Examples:
            >>> from hgp_lib.rules.low_memory_operators import And
            >>> from hgp_lib.rules import Literal
            >>> import numpy as np
            >>> data = np.array([
            ...     [True, False, True],
            ...     [False, False, True]
            ... ])
            >>> rule = Or([Literal(value=0), Literal(value=1)])
            >>> rule.evaluate(data)
            array([ True, False])

        Notes:
            This implementation is efficient and allocates as little memory as possible.
        """
        rez = self.subrules[0].evaluate(data)
        if self.subrules[0].value is not None and not self.subrules[0].negated:
            rez = rez.copy()
        for subrule in self.subrules[1:]:
            rez |= subrule.evaluate(data)
        if self.negated:
            rez = np.logical_not(rez, out=rez)
        return rez
