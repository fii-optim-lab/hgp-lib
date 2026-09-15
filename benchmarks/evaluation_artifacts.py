import json
from pathlib import Path

import numpy as np

from hgp_lib.rules import And, Literal, Or, Rule, deserialize, serialize

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "evaluation_default"
FORMAT_VERSION = 1
NUM_RULES = 100
LITERAL_COUNTS = (100, 1_000)
MAX_DEPTH = 20
SEEDS = {100: 2301, 1_000: 2302}


def artifact_path(num_literals: int) -> Path:
    return ARTIFACT_DIR / f"rules-{num_literals}.json"


def random_literal(rng: np.random.Generator, num_literals: int) -> Literal:
    return Literal(
        value=int(rng.integers(num_literals)),
        negated=bool(rng.integers(2)),
    )


def random_operator(rng: np.random.Generator, subrules: list[Rule]) -> And | Or:
    operator = And if rng.integers(2) == 0 else Or
    return operator(
        subrules,
        negated=bool(rng.integers(2)),
        copy_subrules=False,
    )


def generate_random_rule(rng: np.random.Generator, num_literals: int) -> Rule:
    child_count = int(rng.integers(2, min(6, num_literals) + 1))
    children = [random_literal(rng, num_literals) for _ in range(child_count)]
    root = random_operator(rng, children)
    expandable = [(child, 1) for child in children]
    literal_count = child_count

    while literal_count < num_literals:
        selected = int(rng.integers(len(expandable)))
        literal, depth = expandable.pop(selected)
        remaining = num_literals - literal_count
        child_count = int(rng.integers(2, min(6, remaining + 1) + 1))
        children = [random_literal(rng, num_literals) for _ in range(child_count)]
        operator = random_operator(rng, children)

        parent = literal.parent
        index = next(
            index for index, subrule in enumerate(parent.subrules) if subrule is literal
        )
        operator.parent = parent
        parent.subrules[index] = operator

        child_depth = depth + 1
        if child_depth < MAX_DEPTH:
            expandable.extend((child, child_depth) for child in children)
        literal_count += child_count - 1

    return root


def literal_count(rule: Rule) -> int:
    return sum(isinstance(node, Literal) for node in rule.flatten())


def rule_depth(rule: Rule) -> int:
    depth = 0
    stack = [(rule, 0)]
    while stack:
        node, current_depth = stack.pop()
        depth = max(depth, current_depth)
        stack.extend((child, current_depth + 1) for child in node.subrules)
    return depth


def save_artifact(num_literals: int, rules: list[Rule]) -> None:
    path = artifact_path(num_literals)
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_mapping = {index: f"feature_{index}" for index in range(num_literals)}
    payload = {
        "format_version": FORMAT_VERSION,
        "num_rules": NUM_RULES,
        "num_literals": num_literals,
        "max_depth": MAX_DEPTH,
        "seed": SEEDS[num_literals],
        "rules": [
            serialize(rule, feature_mapping if index == 0 else None)
            for index, rule in enumerate(rules)
        ],
    }

    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))
    temporary_path.replace(path)


def load_artifact(num_literals: int) -> list[Rule]:
    path = artifact_path(num_literals)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format_version": FORMAT_VERSION,
        "num_rules": NUM_RULES,
        "num_literals": num_literals,
        "max_depth": MAX_DEPTH,
        "seed": SEEDS[num_literals],
    }
    for name, value in expected.items():
        if payload[name] != value:
            raise ValueError(
                f"Artifact {path} has {name}={payload[name]!r}, expected {value!r}"
            )

    rules = []
    feature_mapping = None
    for serialized_rule in payload["rules"]:
        rule, current_mapping = deserialize(serialized_rule)
        if current_mapping is not None:
            feature_mapping = current_mapping
        rules.append(rule)

    expected_mapping = {index: f"feature_{index}" for index in range(num_literals)}
    if feature_mapping != expected_mapping:
        raise ValueError(f"Artifact {path} has an incompatible feature mapping")
    if len(rules) != NUM_RULES:
        raise ValueError(f"Artifact {path} must contain {NUM_RULES} rules")
    return rules


def generate_artifact(num_literals: int) -> list[Rule]:
    rng = np.random.default_rng(SEEDS[num_literals])
    rules = [generate_random_rule(rng, num_literals) for _ in range(NUM_RULES)]
    if any(literal_count(rule) != num_literals for rule in rules):
        raise RuntimeError(f"Rules must contain {num_literals} literals")
    if any(rule_depth(rule) > MAX_DEPTH for rule in rules):
        raise RuntimeError(f"Rules must not exceed depth {MAX_DEPTH}")

    save_artifact(num_literals, rules)
    return load_artifact(num_literals)


def create_evaluation_artifacts(overwrite: bool = False) -> None:
    for num_literals in LITERAL_COUNTS:
        path = artifact_path(num_literals)
        if path.exists() and not overwrite:
            print(f"Skipping evaluation rules with {num_literals} literals")
            continue
        print(
            f"Generating evaluation rules with {num_literals} literals...", flush=True
        )
        generate_artifact(num_literals)
        print(f"  rules={NUM_RULES}, literals={num_literals}, max_depth={MAX_DEPTH}")
