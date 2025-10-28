from .rules import Rule


class Literal(Rule):
    """
    Represents a single literal condition in a rule tree.

    A `Literal` corresponds to a single feature (column) in the input `data`.
    It may optionally be negated, in which case its logical value is inverted.

    Attributes:
        subrules (Optional[List[Rule]]):
            The list of child rules, for operators, or `None`, for literals. Default: `None`.
        parent (Optional[Rule]):
            A reference to the parent rule in the tree (if any). Default: `None`.
        value (int):
            The column index of the feature this literal refers to in `data`.
        negated (bool):
            Whether the literal is negated (logical NOT).

    Examples:
        >>> import numpy as np
        >>> data = np.array([[True, False, True], [False, True, False]])
        >>> literal = Literal(value=0)
        >>> literal.evaluate(data)
        array([ True, False])
        >>> negated_literal = Literal(value=1, negated=True)
        >>> negated_literal.evaluate(data)
        array([ True, False])
        >>> str(Literal(value=2))
        '2'
        >>> str(Literal(value=2, negated=True))
        '~2'
    """

    def evaluate(self, data):
        """
        Evaluates this literal on the given data array, based on the `self.value` feature.

        Args:
            data (np.ndarray):
                Input data passed to subrules. Must be a 2D ndarray, with instances on rows and features on columns.
                Not checked at runtime for performance reasons.

        Returns:
            np.ndarray:
                The boolean result of evaluating this rule vectorized across all instances.

        Examples:
            >>> import torch
            >>> data = torch.tensor([[True, False], [False, True]])
            >>> Literal(value=0).evaluate(data)
            tensor([ True, False])
            >>> Literal(value=1, negated=True).evaluate(data)
            tensor([ True, False])
        """
        return ~data[:, self.value] if self.negated else data[:, self.value]

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the literal.

        Returns:
            str: The literal as a string, prefixed with `~` if negated.

        Examples:
            >>> str(Literal(value=0))
            '0'
            >>> str(Literal(value=0, negated=True))
            '~0'
        """
        return f"~{self.value}" if self.negated else f"{self.value}"
