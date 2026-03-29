import random
from typing import Type, Tuple

import numpy as np

from .base_mutation import Mutation
from .utils import MutationError
from ..rules import Rule, Or, And, Literal


class DeleteMutation(Mutation):
    """
    Remove a ``Rule`` node from its parent operator in-place.

    Applicable to both literal and operator nodes. The mutation has two modes
    depending on the parent's subrule count:

    - Parent has **3+ subrules**: the target node is simply removed from the
      parent's subrule list.
    - Parent has **exactly 2 subrules** (collapse): the remaining sibling is
      appended to the grandparent's subrule list and the now-redundant parent
      operator is removed from the grandparent. This requires a grandparent to
      exist; if there is none, a ``MutationError`` is raised.

    A ``MutationError`` is also raised when the target node is the root of the
    tree (has no parent).

    Attributes:
        is_literal_mutation (bool): ``True``.
        is_operator_mutation (bool): ``True``.

    Examples:
        Simple removal (parent has 3 subrules):

        >>> from hgp_lib.mutations import DeleteMutation
        >>> from hgp_lib.rules import And, Literal
        >>> parent = And([Literal(value=0), Literal(value=1), Literal(value=2)])
        >>> mutation = DeleteMutation()
        >>> mutation.apply(parent.subrules[1])
        >>> parent
        And(0, 2)

        Collapse (parent has 2 subrules — sibling is appended to grandparent):

        >>> from hgp_lib.rules import Or
        >>> root = Or([
        ...     Literal(value=0),
        ...     And([Literal(value=1), Literal(value=2)]),
        ... ])
        >>> mutation.apply(root.subrules[1].subrules[0])
        >>> root
        Or(0, 2)
    """

    def __init__(self):
        super().__init__(is_literal_mutation=True, is_operator_mutation=True)

    def apply(self, rule: Rule):
        """
        Delete ``rule`` from its parent's subrule list in-place.

        When the parent has more than two subrules, the target is simply removed.
        When the parent has exactly two subrules, the remaining sibling is appended
        to the grandparent's subrule list and the parent operator is removed from
        the grandparent (collapse).

        Args:
            rule (Rule):
                The rule node to delete.

        Raises:
            MutationError:
                If ``rule`` has no parent (root node), or if the parent has exactly
                two subrules and no grandparent exists.

        Examples:
            Simple removal:

            >>> from hgp_lib.mutations import DeleteMutation
            >>> from hgp_lib.rules import Or, Literal
            >>> parent = Or([Literal(value=1), Literal(value=2), Literal(value=3)])
            >>> DeleteMutation().apply(parent.subrules[2])
            >>> parent
            Or(1, 2)

            Collapse — the inner And is removed and its surviving child is
            appended to the root:

            >>> from hgp_lib.rules import And
            >>> root = Or([
            ...     Literal(value=0),
            ...     And([Literal(value=1), Literal(value=2)]),
            ...     Literal(value=3),
            ... ])
            >>> DeleteMutation().apply(root.subrules[1].subrules[1])
            >>> root
            Or(0, 3, 1)
        """
        parent = rule.parent
        if parent is None:
            raise MutationError()
        subrules = parent.subrules
        if len(subrules) == 2:
            # Special case, we might need to collapse the operator along with the literal
            grandparent = parent.parent
            if grandparent is None:
                raise MutationError()
            other_rule_index = 0
            if subrules[0] is rule:
                other_rule_index = 1
            subrules[other_rule_index].parent = grandparent
            grandparent.subrules.append(subrules[other_rule_index])
            del subrules[1 - other_rule_index]
            subrules = grandparent.subrules
            rule = parent
        for i in range(len(subrules)):
            if subrules[i] is rule:
                del subrules[i]
                return
        raise RuntimeError("Unreachable code")


class NegateMutation(Mutation):
    """
    The `NegateMutation` inverts inplace the logical negation flag of a given `Rule`. It is applicable to both literals
    and operator nodes.

    Attributes:
        is_literal_mutation (bool): `True`.
        is_operator_mutation (bool): `True`.

    Examples:
        >>> from hgp_lib.mutations import NegateMutation
        >>> from hgp_lib.rules import Literal, And
        >>> mutation = NegateMutation()
        >>> rule = Literal(value=1)
        >>> mutation.apply(rule)
        >>> rule
        ~1
        >>> mutation.apply(rule)
        >>> rule
        1
        >>> rule = And([Literal(value=0), Literal(value=1)])
        >>> mutation.apply(rule)
        >>> rule
        ~And(0, 1)
    """

    def __init__(self):
        super().__init__(is_literal_mutation=True, is_operator_mutation=True)

    def apply(self, rule: Rule):
        """
        Applies an inplace negation mutation to the given `Rule`. Toggles the rule's `negated` flag.

        Args:
            rule (Rule):
                The rule node whose negation flag will be flipped.

        Examples:
            >>> from hgp_lib.mutations import NegateMutation
            >>> from hgp_lib.rules import Literal
            >>> rule = Literal(value=0)
            >>> mutation = NegateMutation()
            >>> mutation.apply(rule)
            >>> rule
            ~0
            >>> mutation.apply(rule)
            >>> rule
            0
        """
        rule.negated = not rule.negated


class ReplaceLiteral(Mutation):
    """
    The `ReplaceLiteral` mutation replaces the value of a literal `Rule` with a different random literal index.
    It ensures that the new literal value differs from the current one. This mutation is only applicable to literal
    nodes and never to operator nodes.

    Attributes:
        is_literal_mutation (bool): `True`.
        is_operator_mutation (bool): `False`.
        num_literals (int): The total number of possible literal values. Must be greater than `1`.

    Notes:
        - The new literal value is chosen uniformly at random from the range `[0, num_literals)`.
        - If the randomly chosen value equals the current literal's value, it is incremented modulo `num_literals`
          to guarantee a change.
        - The mutation operates inplace and modifies only the literal's `value`.

    Examples:
        >>> from hgp_lib.mutations import ReplaceLiteral
        >>> from hgp_lib.rules import Literal
        >>> mutation = ReplaceLiteral(num_literals=2)
        >>> rule = Literal(value=0)
        >>> mutation.apply(rule)
        >>> rule
        1
    """

    def __init__(self, num_literals: int):
        super().__init__(is_literal_mutation=True, is_operator_mutation=False)

        # num_literals was validated

        self.num_literals = num_literals

    def apply(self, rule: Rule):
        """
        Applies an inplace literal replacement mutation to the given `Rule`. Randomly assigns a new literal value
        different from the current one.

        Args:
            rule (Rule):
                The literal rule whose value will be replaced.

        Examples:
            >>> from hgp_lib.mutations import ReplaceLiteral
            >>> from hgp_lib.rules import Literal
            >>> mutation = ReplaceLiteral(num_literals=4)
            >>> rule = Literal(value=1)
            >>> mutation.apply(rule)
            >>> rule.value != 1
            True
        """
        new_value = np.random.randint(self.num_literals)
        if new_value == rule.value:
            new_value = (new_value + 1) % self.num_literals
        rule.value = new_value


class PromoteLiteral(Mutation):
    """
    The `PromoteLiteral` mutation converts a literal `Rule` into an operator node (i.e. `And`, `Or`) by promoting it
    and attaching two literal children: one representing the old literal and another newly generated literal.
    This mutation increases the structural complexity of the rule tree.

    Attributes:
        is_literal_mutation (bool): `True`.
        is_operator_mutation (bool): `False`.
        num_literals (int): The total number of possible literal values. Must be greater than `1`.
        operator_types (Tuple[Type[Rule]]): Tuple of operator classes (e.g., `(Or, And)`) that can replace the literal.

    Notes:
        - The literal is transformed inplace into an operator node by changing its class (`__class__`).
        - Two new subrules are attached:
            1. The original literal (same value and negation),
            2. A new literal with a randomly chosen value different from the original.
        - Random negation flags are assigned independently to the new operator and the new literal.
        - The `value` attribute of the promoted node is cleared (`None`) since it becomes an operator.

    Examples:
        >>> from hgp_lib.mutations import PromoteLiteral
        >>> from hgp_lib.rules import Literal, And, Or
        >>> mutation = PromoteLiteral(num_literals=4)
        >>> rule = Literal(value=1, negated=False)
        >>> mutation.apply(rule)
        >>> len(rule)
        3
        >>> isinstance(rule, Or) or isinstance(rule, And)
        True
        >>> rule.subrules[0]
        1
    """

    def __init__(
        self, num_literals: int, operator_types: Tuple[Type[Rule], ...] = (Or, And)
    ):
        super().__init__(is_literal_mutation=True, is_operator_mutation=False)

        # num_literals was validated
        # operator_types was validated

        self.num_literals = num_literals
        self.operator_types = operator_types

    def apply(self, rule: Rule):
        """
        Promotes a literal node into a randomly chosen operator (`And` or `Or`), creating two new literal subrules:
        one representing the old literal and another newly generated literal with a different value.

        Args:
            rule (Rule):
                The literal rule to promote.

        Notes:
            - The transformation occurs inplace by reassigning `rule.__class__`.
            - The mutation randomizes negation for both the operator and the new literal.

        Examples:
            >>> from hgp_lib.mutations import PromoteLiteral
            >>> from hgp_lib.rules import Literal
            >>> mutation = PromoteLiteral(num_literals=3)
            >>> rule = Literal(value=0, negated=True)
            >>> mutation.apply(rule)
            >>> len(rule)
            3
            >>> isinstance(rule, Or) or isinstance(rule, And)
            True
            >>> rule.subrules[0]
            ~0
        """
        rule.__class__ = random.choice(self.operator_types)  # Efficient class change
        negated = (
            np.random.rand(2) < 0.5
        )  # Creating negated flag for the new operator and the new literal
        new_value = np.random.randint(
            self.num_literals
        )  # Creating value for new literal
        if new_value == rule.value:
            new_value = (new_value + 1) % self.num_literals
        rule.subrules = [
            Literal(None, rule, rule.value, rule.negated),  # Old literal
            Literal(None, rule, new_value, negated[0]),  # New literal
        ]
        rule.negated = negated[1]  # Operator negated flag
        rule.value = None  # Removing the value from the new operator
