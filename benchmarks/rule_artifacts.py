import json
import random
from pathlib import Path

import numpy as np

from hgp_lib.algorithms import BooleanGP
from hgp_lib.configs import BooleanGPConfig
from hgp_lib.populations import PopulationGeneratorFactory
from hgp_lib.preprocessing import StandardBinarizer
from hgp_lib.rules import Rule, deserialize, serialize
from hgp_lib.utils import ComplexityCheck

from .data import DATASET_NAMES, N_SPLITS, load_fold

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "rule_evaluation"
FORMAT_VERSION = 2
NUM_GENERATIONS = 500
POPULATION_SIZE = 100
HIGH_COMPLEXITY_REWARD = -0.001
MEDIUM_COMPLEXITY_REWARD = -0.0005
MAX_COMPLEXITIES = (50, 100, 100, 250, 500)
COMPLEXITY_PENALTIES = (
    0.0,
    0.0,
    HIGH_COMPLEXITY_REWARD,
    MEDIUM_COMPLEXITY_REWARD,
    HIGH_COMPLEXITY_REWARD,
)


def artifact_path(dataset: str, fold: int) -> Path:
    return ARTIFACT_DIR / dataset / f"fold-{fold}.json"


def artifact_seed(dataset: str, fold: int) -> int:
    return DATASET_NAMES.index(dataset) * N_SPLITS + fold


def prepare_fold(dataset: str, fold: int):
    X_train, X_test, y_train, y_test = load_fold(dataset, fold)
    binarizer = StandardBinarizer(progress_bar=False)
    train_data = binarizer.fit_transform(X_train, y_train).to_numpy(dtype=bool)
    test_data = binarizer.transform(X_test).to_numpy(dtype=bool)
    feature_mapping = {
        index: str(name) for index, name in enumerate(binarizer.get_feature_names_out())
    }
    return train_data, y_train, test_data, y_test, feature_mapping


def save_artifact(
    dataset: str,
    fold: int,
    rules: list[Rule],
    feature_mapping: dict[int, str],
) -> None:
    path = artifact_path(dataset, fold)
    path.parent.mkdir(parents=True, exist_ok=True)

    complexities = [len(rule) for rule in rules]
    payload = {
        "format_version": FORMAT_VERSION,
        "dataset": dataset,
        "fold": fold,
        "seed": artifact_seed(dataset, fold),
        "num_generations": NUM_GENERATIONS,
        "population_size": POPULATION_SIZE,
        "max_complexity": MAX_COMPLEXITIES[fold],
        "complexity_penalty": COMPLEXITY_PENALTIES[fold],
        "mean_complexity": float(np.mean(complexities)),
        "observed_max_complexity": max(complexities),
        "rules": [
            serialize(rule, feature_mapping if index == 0 else None)
            for index, rule in enumerate(rules)
        ],
    }

    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))
    temporary_path.replace(path)


def load_artifact(dataset: str, fold: int):
    path = artifact_path(dataset, fold)
    payload = json.loads(path.read_text(encoding="utf-8"))

    expected = {
        "format_version": FORMAT_VERSION,
        "dataset": dataset,
        "fold": fold,
        "seed": artifact_seed(dataset, fold),
        "num_generations": NUM_GENERATIONS,
        "population_size": POPULATION_SIZE,
        "max_complexity": MAX_COMPLEXITIES[fold],
        "complexity_penalty": COMPLEXITY_PENALTIES[fold],
    }
    for name, value in expected.items():
        if payload[name] != value:
            raise ValueError(
                f"Artifact {path} has {name}={payload[name]!r}, expected {value!r}"
            )

    serialized_rules = payload["rules"]
    if len(serialized_rules) != POPULATION_SIZE:
        raise ValueError(f"Artifact {path} must contain {POPULATION_SIZE} rules")

    rules = []
    feature_mapping = None
    for serialized_rule in serialized_rules:
        rule, current_mapping = deserialize(serialized_rule)
        if current_mapping is not None:
            if feature_mapping is None:
                feature_mapping = current_mapping
            elif current_mapping != feature_mapping:
                raise ValueError(
                    f"Artifact {path} contains inconsistent feature mappings"
                )
        rules.append(rule)

    if feature_mapping is None:
        raise ValueError(f"Artifact {path} does not contain a feature mapping")
    return rules, feature_mapping, payload


def load_evaluation_artifact(dataset: str, fold: int):
    rules, feature_mapping, metadata = load_artifact(dataset, fold)
    _, _, data, labels, current_mapping = prepare_fold(dataset, fold)
    if feature_mapping != current_mapping:
        raise ValueError(
            f"Artifact {artifact_path(dataset, fold)} has an incompatible feature mapping"
        )
    return rules, data, labels, metadata


def generate_artifact(dataset: str, fold: int):
    train_data, train_labels, _, _, feature_mapping = prepare_fold(dataset, fold)
    seed = artifact_seed(dataset, fold)
    np.random.seed(seed)
    random.seed(seed)

    complexity_check = ComplexityCheck(MAX_COMPLEXITIES[fold])
    config = BooleanGPConfig(
        train_data=train_data,
        train_labels=train_labels,
        population_factory=PopulationGeneratorFactory(population_size=POPULATION_SIZE),
        optimize_scorer=True,
        regeneration=False,
        check_valid=complexity_check,
    )
    gp = BooleanGP(config)
    gp.complexity_penalty = COMPLEXITY_PENALTIES[fold]
    for _ in range(NUM_GENERATIONS):
        gp.step()

    if len(gp.population) != POPULATION_SIZE:
        raise RuntimeError(f"Expected {POPULATION_SIZE} final rules")

    save_artifact(dataset, fold, gp.population, feature_mapping)
    return load_artifact(dataset, fold)


def create_rule_artifacts(overwrite: bool = False) -> None:
    for dataset in DATASET_NAMES:
        for fold in range(N_SPLITS):
            path = artifact_path(dataset, fold)
            if path.exists() and not overwrite:
                print(f"Skipping {dataset} fold {fold}")
                continue

            print(f"Generating {dataset} fold {fold}...", flush=True)
            _, _, metadata = generate_artifact(dataset, fold)
            print(
                f"  rules={metadata['population_size']}, "
                f"mean_complexity={metadata['mean_complexity']:.2f}, "
                f"max_complexity={metadata['observed_max_complexity']}"
            )
