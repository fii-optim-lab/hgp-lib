from typing import Type

from .rules import Rule
from .literals import Literal


def is_operator(op: Rule) -> bool:
    return isinstance(op, Rule) and not isinstance(op, Literal)


def is_operator_type(t: Type[Rule]) -> bool:
    return isinstance(t, type) and issubclass(t, Rule) and not issubclass(t, Literal)


def replace_with_rule(target: Rule, rule: Rule) -> None:
    """
    Replaces the content of `target` with the content of `rule` in-place.

    This function mutates `target` to have the same class, value, negation state,
    and subrules as `rule`. The subrules are deep-copied with `target` as their parent.

    Args:
        target (Rule): The rule whose content will be replaced.
        rule (Rule): The rule whose content will be copied into `target`.

    Examples:
        >>> from hgp_lib.rules import And, Or, Literal
        >>> target = And([Literal(value=0), Literal(value=1)])
        >>> source = Or([Literal(value=2), Literal(value=3)])
        >>> replace_with_rule(target, source)
        >>> type(target).__name__
        'Or'
        >>> str(target)
        'Or(2, 3)'
    """
    target.__class__ = rule.__class__
    target.value = rule.value
    target.subrules = [s.copy(target) for s in rule.subrules]
    target.negated = rule.negated


def deep_swap(node_a: Rule, node_b: Rule) -> None:
    """
    Swaps the content of two `Rule` nodes in-place.

    Both nodes are mutated so that each takes on the class, value, negation state,
    and subrules of the other. This is useful for subtree crossover operations.

    Args:
        node_a (Rule): First node to swap.
        node_b (Rule): Second node to swap.

    Examples:
        >>> from hgp_lib.rules import And, Or, Literal
        >>> node_a = And([Literal(value=0), Literal(value=1)])
        >>> node_b = Or([Literal(value=2), Literal(value=3)])
        >>> deep_swap(node_a, node_b)
        >>> str(node_a)
        'Or(2, 3)'
        >>> str(node_b)
        'And(0, 1)'
    """
    copy_node_a = node_a.copy()
    copy_node_b = node_b.copy()
    replace_with_rule(node_a, copy_node_b)
    replace_with_rule(node_b, copy_node_a)
