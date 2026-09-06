import random

import numpy as np
import pytest

from hgp_lib import BooleanGPConfig, BooleanRuleClassifier, TrainerConfig
from hgp_lib.utils.metrics import fast_f1_score

from .data import DATASET_NAMES, N_SPLITS, load_fold

NUM_EPOCHS = 500


def scenario_id(dataset: str) -> str:
    return f"full_run.{dataset}.{NUM_EPOCHS}_epochs"


def dataset_parameter(dataset: str):
    identifier = scenario_id(dataset)
    marks = [
        pytest.mark.scenario(identifier),
        pytest.mark.benchmark(
            group=identifier,
            disable_gc=True,
            warmup=False,
        ),
    ]
    return pytest.param(dataset, id=dataset, marks=marks)


@pytest.mark.parametrize(
    "dataset",
    [dataset_parameter(dataset) for dataset in DATASET_NAMES],
)
@pytest.mark.parametrize(
    "fold",
    range(N_SPLITS),
    ids=lambda fold: f"fold-{fold}",
)
def test_full_run(benchmark, dataset, fold):
    X_train, X_test, y_train, y_test = load_fold(dataset, fold)
    config = TrainerConfig(
        gp_config=BooleanGPConfig(),
        num_epochs=NUM_EPOCHS,
        val_every=100,
        progress_bar=False,
    )

    def fit_classifier():
        classifier = BooleanRuleClassifier(config)
        classifier.fit(X_train, y_train, X_test, y_test)
        return classifier

    np.random.seed(fold)
    random.seed(fold)
    classifier = benchmark.pedantic(
        fit_classifier,
        rounds=2,
        iterations=2,
    )

    predictions = classifier.predict(X_test)
    test_score = float(fast_f1_score(y_test, predictions))
    benchmark.extra_info.update(
        {
            "scenario_id": scenario_id(dataset),
            "case_id": f"fold-{fold}",
            "dataset": dataset,
            "fold": fold,
            "num_epochs": NUM_EPOCHS,
            "test_score": test_score,
        }
    )
