from .rules import Rule


class ComplexityCheck:
    """
    Create a validity predicate that rejects rules exceeding ``max_complexity`` nodes.

    Intended for use as the ``check_valid`` argument of ``BooleanGPConfig``.

    Args:
        max_complexity (int):
            Maximum allowed node count. Default: `100`.

    Examples:
        >>> from hgp_lib.rules import And, ComplexityCheck, Literal
        >>> check = ComplexityCheck(3)
        >>> check(Literal(value=0))
        True
        >>> check(And([Literal(value=0), Literal(value=1)]))
        True
        >>> check(And([Literal(value=0), And([Literal(value=1), Literal(value=2)])]))
        False
    """

    def __init__(self, max_complexity: int = 100):
        self.max_complexity = max_complexity

    def __call__(self, rule: Rule) -> bool:
        """
        Check if rule complexity (node count) is within a limit.

        Args:
            rule (Rule): The rule to check.

        Returns:
            bool: ``True`` if ``len(rule) <= self.max_complexity``.

        Examples:
            >>> from hgp_lib.rules import And, ComplexityCheck, Literal
            >>> ComplexityCheck(5)(Literal(value=0))
            True
            >>> ComplexityCheck(2)(And([Literal(value=0), Literal(value=1)]))
            False
        """
        return len(rule) <= self.max_complexity
