from typing import Dict, Optional


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
            >>> import numpy as np
            >>> data = np.array([[True, False], [False, True]])
            >>> Literal(value=0).evaluate(data)
            array([ True, False])
            >>> Literal(value=1, negated=True).evaluate(data)
            array([ True, False])
        """
        return ~data[:, self.value] if self.negated else data[:, self.value]

    def to_str(
        self, feature_names: Dict[int, str] | None = None, indent: bool = -1
    ) -> str:
        """
        Returns a human-readable string representation of the literal. The literal can be replaced with the feature
        name if provided.

        Args:
            feature_names (Dict[int, str] | None): The feature names that can be used to replace literal values when
                provided. Default: `None`.
            indent (int): Not used. Default: `-1`.

        Returns:
            str: The literal as a string, prefixed with `~` if negated.

        Examples:
            >>> str(Literal(value=0))
            '0'
            >>> str(Literal(value=0, negated=True))
            '~0'
            >>> Literal(value=0, negated=True).to_str({0: "bad"})
            '~bad'
        """
        value = self.value
        if feature_names is not None:
            value = feature_names[value]
        return f"~{value}" if self.negated else f"{value}"

    def copy(self, parent: Optional["Rule"] = None) -> "Rule":
        """
        Creates a copy of this Literal, optionally assigning a new parent.
        This uses a faster execution path to create literals.

        Args:
            parent (Optional[Rule]):
                The parent rule for the new copy. If omitted, retains the current parent. Default: `None`.

        Returns:
            Rule: A copy of the current Literal.

        Examples:
            >>> from hgp_lib.rules import And, Literal
            >>> a = Literal(value=1)
            >>> b = a.copy()
            >>> a is b
            False
        """
        return Literal(
            None,
            self.parent if parent is None else parent,
            self.value,
            self.negated,
            False,
        )
