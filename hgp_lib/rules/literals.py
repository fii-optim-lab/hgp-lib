from .rules import Rule


class Literal(Rule):
    """
    Represents a single literal condition in a rule tree.

    A `Literal` corresponds to a single feature (column) in the input `data`.
    It may optionally be negated, in which case its logical value is inverted.

    Attributes:
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
        Evaluates this literal on the given data array.

        Args:
            data: A 2D array where each column represents a feature.

        Returns:
            xp.ndarray: Boolean array corresponding to this literal’s value
                        (negated if `self.negated` is True).

        Example:
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
