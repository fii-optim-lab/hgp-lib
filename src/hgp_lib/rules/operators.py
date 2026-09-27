from .rules import Rule


class And(Rule):
    """
    Logical conjunction (`AND`) operator node for rule trees. It evaluates to `True` only if every subrule evaluates to
    `True`.

    Attributes:
        subrules (list[Rule]):
            A list of child rules combined with logical AND. Must be a list longer than 1 element. Not checked at
            runtime for performance reasons. Default: `None`.
        parent (Rule | None):
            A reference to the parent rule, if part of a larger tree. Default: `None`.
        value (None):
            Always `None` for operator nodes (non-literals). Not checked at runtime for performance reasons. Default:
            `None`.
        negated (bool):
            Whether the entire conjunction is logically negated (`~And(...)`). Default: `False`.

    Examples:
        >>> from hgp_lib.rules import And, Literal, Or
        >>> rule = And([
        ...     Literal(value=0),
        ...     Or([Literal(value=1, negated=True), Literal(value=2)]),
        ...     Literal(value=3)
        ... ])
        >>> rule
        And(0, Or(~1, 2), 3)
    """


class Or(Rule):
    """
    Logical disjunction (`OR`) operator node for rule trees. It evaluates to `True` if any subrule evaluates to `True`.

    Attributes:
        subrules (list[Rule]):
            A list of child rules combined with logical OR. Must be a list longer than 1 element. Not checked at
            runtime for performance reasons. Default: `None`.
        parent (Rule | None):
            A reference to the parent rule, if part of a larger tree. Default: `None`.
        value (None):
            Always `None` for operator nodes (non-literals). Not checked at runtime for performance reasons. Default:
            `None`.
        negated (bool):
            Whether the entire disjunction is logically negated (`~Or(...)`). Default: `False`.

    Examples:
        >>> from hgp_lib.rules import Literal, Or
        >>> rule = Or([Literal(value=0), Literal(value=1, negated=True)])
        >>> rule
        Or(0, ~1)
    """


# TODO: Add higher level operators from Boolxai
