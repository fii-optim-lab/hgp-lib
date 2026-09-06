import random

from .literals import Literal
from .rules import Rule


def is_operator(op: Rule) -> bool:
    """
    Check whether a rule node is an operator (non-literal).

    Args:
        op (Rule):
            The rule node to check.

    Returns:
        bool: ``True`` if ``op`` is a ``Rule`` but not a ``Literal``.

    Examples:
        >>> from hgp_lib.rules import And, Or, Literal
        >>> from hgp_lib.rules.utils import is_operator
        >>> is_operator(And([Literal(value=0), Literal(value=1)]))
        True
        >>> is_operator(Literal(value=0))
        False
    """
    return isinstance(op, Rule) and not isinstance(op, Literal)


def is_operator_type(t: type[Rule]) -> bool:
    """
    Check whether a type is an operator class (a ``Rule`` subclass that is not ``Literal``).

    Args:
        t (type[Rule]):
            The type to check.

    Returns:
        bool: ``True`` if ``t`` is a subclass of ``Rule`` but not of ``Literal``.

    Examples:
        >>> from hgp_lib.rules import And, Or, Literal
        >>> from hgp_lib.rules.utils import is_operator_type
        >>> is_operator_type(And)
        True
        >>> is_operator_type(Literal)
        False
        >>> is_operator_type(str)
        False
    """
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

    # We don't need full copy, rule was already copied once
    for s in rule.subrules:
        s.parent = target
    target.subrules = rule.subrules

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
    copy_node_a = node_a.detach_subtree()
    copy_node_b = node_b.detach_subtree()
    replace_with_rule(node_a, copy_node_b)
    replace_with_rule(node_b, copy_node_a)


def apply_feature_mapping(rule: Rule, feature_mapping: dict[int, int] | None) -> Rule:
    """
    Creates a copy of a rule with feature indices remapped according to the provided mapping.

    This is a convenience wrapper that handles the common case of optionally applying
    a feature mapping. If no mapping is provided, the original rule is returned unchanged.
    Otherwise, a deep copy is made and the mapping is applied in-place to the copy.

    Args:
        rule (Rule): The rule to potentially remap.
        feature_mapping (dict[int, int] | None): A dictionary mapping old feature indices
            to new feature indices, or None to skip remapping.

    Returns:
        Rule: The original rule if `feature_mapping` is None, otherwise a new copy
            with remapped feature indices.

    Examples:
        >>> from hgp_lib.rules import And, Literal
        >>> from hgp_lib.rules.utils import apply_feature_mapping
        >>> rule = And([Literal(value=0), Literal(value=1)])
        >>> mapped = apply_feature_mapping(rule, {0: 5, 1: 10})
        >>> str(mapped)
        'And(5, 10)'
        >>> str(rule)  # Original unchanged
        'And(0, 1)'
        >>> apply_feature_mapping(rule, None) is rule
        True
    """
    if feature_mapping is None:
        return rule
    new_rule = rule.copy()
    new_rule.apply_feature_mapping(feature_mapping)
    return new_rule


def select_crossover_point(rule: Rule, operator_p: float = 0.9) -> Rule:
    """
    Selects a random node from the rule tree using Koza-style biased sampling.
    This method favors internal operator nodes (e.g., `And`, `Or`) over terminal
    literal nodes (e.g., `Literal`) based on the specified probability, promoting
    structural crossover over simple point mutation.
    Args:
        rule (Rule):
            The root of the rule tree from which to select a node.
        operator_p (float):
            The probability of selecting an internal operator node. If the tree contains
            both operators and literals, operators are chosen with this probability.
            Default: `0.9`.
    Returns:
        Rule:
            A reference to the selected node (either an operator or a literal).
    Notes:
        - Uses a two-way reservoir sampling algorithm to perform selection in a single
          pass (O(N)) with constant memory overhead, avoiding the need to flatten the tree.
        - If the tree consists of only one type of node (e.g., a single Literal),
          that node is returned regardless of `func_prob`.
    Examples:
        >>> import random
        >>> from hgp_lib.rules.utils import select_crossover_point
        >>> from hgp_lib.rules import And, Or, Literal
        >>> random.seed(42)
        >>> rule = And([Literal(value=0), Or([Literal(value=1), Literal(value=2)])])
        >>> selected = select_crossover_point(rule, operator_p=1.0)
        >>> isinstance(selected, (And, Or))
        True
        >>> selected = select_crossover_point(rule, operator_p=0.0)
        >>> isinstance(selected, Literal)
        True
    """
    # TODO: Perf against a flatten in operators and literals list, and 2 random calls.
    selected_operator = selected_literal = None
    count_operator = count_literal = 0

    stack = [rule]
    while stack:
        new_stack = []
        for current in stack:
            if current.subrules:
                count_operator += 1
                if random.random() < (1.0 / count_operator):
                    selected_operator = current
                new_stack.extend(current.subrules)
            else:
                count_literal += 1
                if random.random() < (1.0 / count_literal):
                    selected_literal = current
        stack = new_stack

    if selected_operator and random.random() < operator_p:
        return selected_operator
    return selected_literal


def _create_unsafe_rule(
    rule_type: type[Rule], subrules: list[Rule], parent: Rule, value: int, negated: bool
) -> Rule:
    """
    Creates a new rule while skipping subrules assignment and constructor validation. To be used internally.

    Attributes:
        subrules (list[Rule] | None):
            The list of child rules, for operators, or `None`, for literals. Default: `None`.
        parent (Rule | None):
            A reference to the parent rule in the tree (if any). Default: `None`.
        value (int | None):
            The value held by this rule (e.g., for literals). Should be `None` for operators. Default: `None`.
        negated (bool):
            Whether this rule or literal is logically negated (e.g., `~A`). Default: `False`.

    """
    new = object.__new__(rule_type)
    new.subrules = subrules
    new.parent = parent
    new.value = value
    new.negated = negated
    return new


def serialize(
    rule: Rule, feature_mapping: dict[int, str] | None = None
) -> dict[str, object]:
    """
    Serialize a rule and optional feature names into JSON-compatible data.

    Args:
        rule (Rule): Rule tree to serialize.
        feature_mapping (dict[int, str] | None): Optional mapping from literal indices
            to feature names. Default: `None`.

    Returns:
        dict[str, object]: Serialized feature mapping and rule tree.

    Raises:
        TypeError: If the rule type or feature mapping types are unsupported.
        ValueError: If a feature mapping index is negative.

    Examples:
        >>> from hgp_lib.rules import And, Literal
        >>> from hgp_lib.rules.utils import serialize
        >>> rule = And([Literal(value=0), Literal(value=1, negated=True)])
        >>> serialized = serialize(rule, {0: "age", 1: "income"})
        >>> serialized["feature_mapping"]
        {0: 'age', 1: 'income'}
        >>> serialized["rule"]
        {'And': {'negated': False, 'subrules': [{'Literal': {'negated': False, 'value': 0}}, {'Literal': {'negated': True, 'value': 1}}]}}
    """
    if not isinstance(rule, Rule):
        raise TypeError(f"rule must be a Rule, is {type(rule)}")
    return {
        "feature_mapping": _serialize_feature_mapping(feature_mapping),
        "rule": _serialize_node(rule),
    }


def deserialize(
    serialized_rule: dict[str, object],
) -> tuple[Rule, dict[int, str] | None]:
    """
    Deserialize a rule and its optional feature-name mapping.

    String mapping keys produced by JSON decoding are converted back to integers.

    Args:
        serialized_rule (dict[str, object]): Data produced by :func:`serialize`.

    Returns:
        tuple[Rule, dict[int, str] | None]: The reconstructed rule and feature mapping.

    Raises:
        TypeError: If a serialized value has an invalid type.
        ValueError: If the serialized structure or a mapping index is invalid.

    Examples:
        >>> from hgp_lib.rules import Or, Literal
        >>> from hgp_lib.rules.utils import deserialize, serialize
        >>> original = Or([Literal(value=0), Literal(value=2, negated=True)])
        >>> restored, feature_mapping = deserialize(
        ...     serialize(original, {0: "age", 2: "income"})
        ... )
        >>> str(restored)
        'Or(0, ~2)'
        >>> feature_mapping
        {0: 'age', 2: 'income'}
        >>> all(child.parent is restored for child in restored.subrules)
        True
    """
    if not isinstance(serialized_rule, dict):
        raise TypeError("serialized_rule must be a dictionary")
    if set(serialized_rule) != {"feature_mapping", "rule"}:
        raise ValueError("serialized_rule must contain feature_mapping and rule")

    feature_mapping = _deserialize_feature_mapping(serialized_rule["feature_mapping"])
    rule = _deserialize_node(serialized_rule["rule"])
    return rule, feature_mapping


def _serialize_feature_mapping(
    feature_mapping: dict[int, str] | None,
) -> dict[int, str] | None:
    if feature_mapping is None:
        return None
    if not isinstance(feature_mapping, dict):
        raise TypeError("feature_mapping must be a dictionary or None")

    result = {}
    for index, name in feature_mapping.items():
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("feature mapping indices must be integers")
        if index < 0:
            raise ValueError("feature mapping indices must be non-negative")
        if not isinstance(name, str):
            raise TypeError("feature mapping names must be strings")
        result[index] = name
    return result


def _deserialize_feature_mapping(
    feature_mapping: object,
) -> dict[int, str] | None:
    if feature_mapping is None:
        return None
    if not isinstance(feature_mapping, dict):
        raise TypeError("feature_mapping must be a dictionary or None")

    result = {}
    for raw_index, name in feature_mapping.items():
        if isinstance(raw_index, bool):
            raise TypeError("feature mapping indices must be integers")
        if isinstance(raw_index, int):
            index = raw_index
        elif isinstance(raw_index, str):
            try:
                index = int(raw_index)
            except ValueError as error:
                raise ValueError(
                    f"Invalid feature mapping index: {raw_index!r}"
                ) from error
        else:
            raise TypeError("feature mapping indices must be integers")
        if index < 0:
            raise ValueError("feature mapping indices must be non-negative")
        if not isinstance(name, str):
            raise TypeError("feature mapping names must be strings")
        if index in result:
            raise ValueError(f"Duplicate feature mapping index: {index}")
        result[index] = name
    return result


def _serialize_node(rule: Rule) -> dict[str, dict[str, object]]:
    from .low_memory_operators import And as LowMemoryAnd
    from .low_memory_operators import Or as LowMemoryOr
    from .operators import And as StandardAnd
    from .operators import Or as StandardOr

    if isinstance(rule, Literal):
        return {
            "Literal": {
                "negated": bool(rule.negated),
                "value": int(rule.value),
            }
        }
    if isinstance(rule, (StandardAnd, LowMemoryAnd)):
        name = "And"
    elif isinstance(rule, (StandardOr, LowMemoryOr)):
        name = "Or"
    else:
        raise TypeError(f"Unsupported rule type: {type(rule).__name__}")
    return {
        name: {
            "negated": bool(rule.negated),
            "subrules": [_serialize_node(subrule) for subrule in rule.subrules],
        }
    }


def _deserialize_node(serialized_node: object) -> Rule:
    from . import And, Or

    if not isinstance(serialized_node, dict):
        raise TypeError("A serialized rule node must be a dictionary")
    if len(serialized_node) != 1:
        raise ValueError("A serialized rule node must contain exactly one rule type")

    name, attributes = next(iter(serialized_node.items()))
    if not isinstance(attributes, dict):
        raise TypeError("Serialized rule attributes must be a dictionary")

    if name == "Literal":
        if set(attributes) != {"negated", "value"}:
            raise ValueError("Literal must contain negated and value")
        negated = attributes["negated"]
        value = attributes["value"]
        if not isinstance(negated, bool):
            raise TypeError("Literal negated must be boolean")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Literal value must be an integer")
        if value < 0:
            raise ValueError("Literal value must be non-negative")
        return Literal(value=value, negated=negated)

    if name not in {"And", "Or"}:
        raise ValueError(f"Unknown rule type: {name!r}")
    if set(attributes) != {"negated", "subrules"}:
        raise ValueError(f"{name} must contain negated and subrules")
    negated = attributes["negated"]
    subrules = attributes["subrules"]
    if not isinstance(negated, bool):
        raise TypeError(f"{name} negated must be boolean")
    if not isinstance(subrules, list):
        raise TypeError(f"{name} subrules must be a list")
    if not subrules:
        raise ValueError(f"{name} must contain at least one subrule")

    rule_type = And if name == "And" else Or
    return rule_type(
        [_deserialize_node(subrule) for subrule in subrules],
        negated=negated,
        copy_subrules=False,
    )
