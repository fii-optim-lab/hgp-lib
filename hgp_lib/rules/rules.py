# Reimplementation based on https://github.com/fidelity/boolxai/blob/main/boolxai/rules/rule.py
from abc import ABC, abstractmethod
from typing import List, Optional, Dict

import numpy as np


class Rule(ABC):
    """
    Abstract base class for logical rule nodes used in rule trees.

    Each `Rule` represents either:
    - An `operator`: a logical operation combining subrules (e.g., `And`, `Or`), or
    - A `literal`: a literal condition (e.g., `Literal(value=5, negated=True)`).

    Rules can be nested to form complex logical expressions.
    The tree can be traversed, copied, or evaluated against data.

    Attributes:
        subrules (Optional[List[Rule]]):
            The list of child rules, for operators, or `None`, for literals. Default: `None`.
        parent (Optional[Rule]):
            A reference to the parent rule in the tree (if any). Default: `None`.
        value (Optional[int]):
            The value held by this rule (e.g., for literals). Should be `None` for operators. Default: `None`.
        negated (bool):
            Whether this rule or literal is logically negated (e.g., `~A`). Default: `False`.
        copy_subrules (bool):
            Whether to deep copy the subrules (valid only for operators). If False, the subrules are moved.
            Default: `True`.

    Notes:
        - `__slots__` are used for performance optimization to reduce memory overhead.
        - No runtime validation is performed for speed; incorrect usage may cause undefined behavior.

    Examples:
        >>> from hgp_lib.rules import And, Or, Literal
        >>> rule = And([
        ...     Literal(value=0),
        ...     Or([Literal(value=1, negated=True), Literal(value=2)]),
        ...     Literal(value=3)
        ... ])
        >>> rule
        And(0, Or(~1, 2), 3)
    """

    __slots__ = ("subrules", "parent", "value", "negated")

    def __init__(
        self,
        subrules: Optional[List["Rule"]] = None,
        parent: Optional["Rule"] = None,
        value: Optional[int] = None,
        negated: bool = False,
        copy_subrules: bool = True,
    ):
        self.subrules = []
        if subrules is not None:
            self.extend(subrules, copy=copy_subrules)

        self.parent = parent
        self.value = value
        self.negated = negated

    def extend(self, new_subrules: List["Rule"], copy: bool = True):
        """
        Extends the current subrules with a list of new subrules.

        Args:
            new_subrules (List[Rule]): The list of new subrules to add.
            copy (bool): If True, deep copies each new subrule. If False, moves them (assigns parent to self).
                         Default: True.
        """
        # TODO: Check where this can be added to improve performance.
        if copy:
            self.subrules.extend([s.copy(self) for s in new_subrules])
        else:
            for s in new_subrules:
                s.parent = self
            self.subrules.extend(new_subrules)

    def flatten(self) -> List["Rule"]:
        """
        Recursively flattens the rule subtree into a single list of all `Rule` nodes  using a preorder traversal.

        Returns:
            List[Rule]: A flat list containing `self` followed by all descendant rules in preorder sequence.

        Examples:
            >>> from hgp_lib.rules import And, Or, Literal
            >>> rule = And([
            ...     Literal(value=0),
            ...     Or([Literal(value=1, negated=True), Literal(value=2)]),
            ...     Literal(value=3)
            ... ])
            >>> rule.flatten()
            [And(0, Or(~1, 2), 3), 0, Or(~1, 2), ~1, 2, 3]
        """
        result = [self]
        for subrule in self.subrules:
            result.extend(subrule.flatten())
        return result

    def __len__(self) -> int:
        """
        Returns the total number of nodes in this rule subtree, including the current rule and all its descendants.

        Returns:
            int: The total number of `Rule` nodes in this subtree.

        Examples:
            >>> from hgp_lib.rules import And, Or, Literal
            >>> len(Literal(value=1))
            1
            >>> len(Or([Literal(value=2), Literal(value=3)]))
            3
            >>> len(And([Literal(value=1), Or([Literal(value=2), Literal(value=3)])]))
            5
        """
        return 1 + sum([len(s) for s in self.subrules])

    def to_str(
        self, feature_names: Dict[int, str] | None = None, indent: int = -1
    ) -> str:
        """
        Returns a human-readable string representation of this rule and replaces the literal values with the feature
        names if available.

        Args:
            feature_names (Dict[int, str] | None): The feature names that can be used to replace literal values when
                provided. Default: `None`.
            indent (int): The indentation level when printing the rules. If `-1`, no indentation is used.
                For standard indentation, use `0`. Default: `-1`.

        Returns:
            str: A string representation such as `And(A, B)` or `Literal(1)`.

        Examples:
            >>> from hgp_lib.rules import And, Literal
            >>> str(And([Literal(value=1), Literal(value=2)]))
            'And(1, 2)'
            >>> str(And([Literal(value=1), Literal(value=2)], negated=True))
            '~And(1, 2)'
            >>> And([Literal(value=1), Literal(value=2)], negated=True).to_str({1: "good", 2:"nice"})
            '~And(good, nice)'
        """
        if indent == -1:
            rez = f"{type(self).__name__}({', '.join(s.to_str(feature_names, indent) for s in self.subrules)})"
        else:
            new_indent = indent + 1
            separator = ",\n"
            tab = "\t"
            to_join = separator.join(
                tab * new_indent + s.to_str(feature_names, new_indent)
                for s in self.subrules
            )
            rez = f"{type(self).__name__}(\n{to_join}\n{tab * indent})"

        if self.negated:
            return "~" + rez
        return rez

    def __str__(self) -> str:
        return self.to_str()

    def __repr__(self) -> str:
        return self.to_str()

    def copy(self, parent: Optional["Rule"] = None) -> "Rule":
        """
        Creates a deep copy of this rule and its entire subtree, optionally assigning a new parent.

        Args:
            parent (Optional[Rule]):
                The parent rule for the new copy. If omitted, retains the current parent. Default: `None`.

        Returns:
            Rule: A new instance of the same rule type, with all subrules recursively copied.

        Examples:
            >>> from hgp_lib.rules import And, Literal
            >>> a = And([Literal(value=1), Literal(value=2)])
            >>> b = a.copy()
            >>> a is b
            False
            >>> a.subrules[0] is b.subrules[0]
            False
            >>> all([(x.value is None and y.value is None) or (x.value == y.value) for x, y in zip(a.flatten(), b.flatten())])
            True
        """

        return self.__class__(
            self.subrules,
            self.parent if parent is None else parent,
            self.value,
            self.negated,
        )

    @abstractmethod
    def evaluate(self, data: np.ndarray) -> np.ndarray:
        """
        Abstract method to evaluate this rule against the given data, in a vectorized manner.

        Args:
            data (np.ndarray): The input data. Must be a 2D ndarray, with instances on rows and features on columns.

        Returns:
            np.ndarray:
                The boolean result of evaluating this rule vectorized across all instances.

        Notes:
            Concrete subclasses (`And`, `Or`, `Literal`, etc.) must implement this.
        """
        pass

    def apply_feature_mapping(self, feature_mapping: Dict[int, int]):
        """
        Applies a feature mapping to this rule and all its subrules in-place.

        This method remaps feature indices used in literals according to the provided
        mapping dictionary. It is used in hierarchical GP when child populations operate
        on a subset of features (feature bagging) and need to be translated back to the
        parent's feature space during crossover.

        Args:
            feature_mapping (Dict[int, int]):
                A dictionary mapping old feature indices to new feature indices.
                For literals, `self.value` is replaced with `feature_mapping[self.value]`.

        Returns:
            None: This method modifies the rule in-place.

        Raises:
            KeyError: If `self.value` is not found in `feature_mapping`.

        Examples:
            >>> from hgp_lib.rules import And, Literal
            >>> rule = And([Literal(value=0), Literal(value=1)])
            >>> rule.apply_feature_mapping({0: 5, 1: 10})
            >>> str(rule)
            'And(5, 10)'
        """
        if self.value is not None:
            self.value = feature_mapping[self.value]
        else:
            for subrule in self.subrules:
                subrule.apply_feature_mapping(feature_mapping)
