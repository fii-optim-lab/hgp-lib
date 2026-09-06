import random

import numpy as np
import pytest

from hgp_lib.algorithms import BooleanGP
from hgp_lib.configs import BooleanGPConfig
from hgp_lib.populations import PopulationGeneratorFactory
from hgp_lib.utils.metrics import fast_f1_score

from .data import DATASET_NAMES, N_SPLITS
from .rule_artifacts import POPULATION_SIZE, load_evaluation_artifact


def scenario_id(dataset: str) -> str:
    return f"rule_evaluation.{dataset}.{POPULATION_SIZE}_rules"


def dataset_parameter(dataset: str):
    identifier = scenario_id(dataset)
    return pytest.param(
        dataset,
        id=dataset,
        marks=[
            pytest.mark.fast,
            pytest.mark.scenario(identifier),
            pytest.mark.benchmark(
                group=identifier,
                disable_gc=True,
                warmup=False,
            ),
        ],
    )


@pytest.mark.parametrize(
    "dataset",
    [dataset_parameter(dataset) for dataset in DATASET_NAMES],
)
def test_rule_evaluation(benchmark, dataset):
    evaluators = []
    metadata_per_fold = []
    repeats = 25

    for fold in range(N_SPLITS):
        rules, data, labels, metadata = load_evaluation_artifact(dataset, fold)
        np.random.seed(metadata["seed"])
        random.seed(metadata["seed"])
        gp = BooleanGP(
            BooleanGPConfig(
                train_data=data,
                train_labels=labels,
                population_factory=PopulationGeneratorFactory(
                    population_size=POPULATION_SIZE
                ),
                optimize_scorer=False,
                check_valid=None,
            )
        )
        gp.population = rules
        gp.population_size = len(rules)

        data = np.tile(data, (repeats, 1))
        labels = np.tile(labels, repeats)
        evaluators.append((gp, data, labels))
        metadata_per_fold.append(metadata)

    def evaluate_folds():
        ret = [
            gp.evaluate_population(data, labels, fast_f1_score)
            for gp, data, labels in evaluators
        ]
        for _ in range(repeats):
            [
                gp.evaluate_population(data, labels, fast_f1_score)
                for gp, data, labels in evaluators
            ]
        return ret

    scores_per_fold = benchmark(evaluate_folds)
    if any(len(scores) != POPULATION_SIZE for scores in scores_per_fold):
        raise RuntimeError(f"Expected {POPULATION_SIZE} scores per fold")

    fold_scores = [float(np.mean(scores)) for scores in scores_per_fold]
    benchmark.extra_info.update(
        {
            "scenario_id": scenario_id(dataset),
            "case_id": "all-folds",
            "dataset": dataset,
            "num_folds": N_SPLITS,
            "population_size": POPULATION_SIZE,
            "fold_scores": fold_scores,
            "test_score": float(np.mean(fold_scores)),
            "test_score_std": float(np.std(fold_scores)),
            "mean_rule_complexity": float(
                np.mean([metadata["mean_complexity"] for metadata in metadata_per_fold])
            ),
            "max_rule_complexity": max(
                metadata["observed_max_complexity"] for metadata in metadata_per_fold
            ),
        }
    )
