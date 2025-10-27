import numpy as np

from .rules import Rule


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
        >>> from hgp_lib.rules.operators import And, Or
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
            >>> from hgp_lib.rules.operators import And
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
            This implementation does as few operations as possible, at the expense of more memory usage.
        """
        cols = []
        neg_mask = []
        sub_operators = []
        for s in self.subrules:
            if s.value is not None:  # We have a literal
                cols.append(s.value)
                neg_mask.append(s.negated)
            else:  # We have an operator
                sub_operators.append(s)

        if cols:  # Hot branch for literals
            mask = (data[:, cols] ^ np.array(neg_mask)).all(1)  # One-liner for all literals
            # Updating with operators
            for s in sub_operators:
                mask &= s.evaluate(data)
        else:  # Hot branch for no literals
            mask = sub_operators[0].evaluate(data)  # Create an initial matrix
            for s in sub_operators[1:]:  # Updating with the rest of the operators
                mask &= s.evaluate(data)

        if self.negated:
            mask = np.logical_not(mask, out=mask)
        return mask


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
        >>> from hgp_lib.rules.operators import And
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
            >>> from hgp_lib.rules.operators import And
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
            This implementation does as few operations as possible, at the expense of more memory usage.
        """
        cols = []
        neg_mask = []
        sub_operators = []
        for s in self.subrules:
            if s.value is not None:  # We have a literal
                cols.append(s.value)
                neg_mask.append(s.negated)
            else:  # We have an operator
                sub_operators.append(s)

        if cols:  # Hot branch for literals
            mask = (data[:, cols] ^ np.array(neg_mask)).any(1)  # One-liner for all literals
            # Updating with operators
            for s in sub_operators:
                mask |= s.evaluate(data)
        else:  # Hot branch for no literals
            mask = sub_operators[0].evaluate(data)  # Create an initial matrix
            for s in sub_operators[1:]:  # Updating with the rest of the operators
                mask |= s.evaluate(data)

        if self.negated:
            mask = np.logical_not(mask, out=mask)
        return mask

# TODO: Add higher level operators from Boolxai
