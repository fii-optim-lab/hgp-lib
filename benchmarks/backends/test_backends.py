"""
Backend benchmarks: the same scenarios for every option combination of every backend.

Run them with ``python benchmarks/benchmark_backends.py --machine <machine> --version
<version>``. They are not part of the default benchmark, which ignores this directory.

Every scenario records its result (a mean score) as ``test_score``. Backend options only
change speed and memory use, so the result must be the same for every combination; the
report flags scenarios where it is not.
"""

import itertools
import random
from dataclasses import asdict
from functools import lru_cache

import numpy as np
import pytest

from hgp_lib.algorithms import BooleanGP
from hgp_lib.configs import BooleanGPConfig
from hgp_lib.evaluation import Dataset, NumpyBackend, Scorer, fast_f1_score
from hgp_lib.populations import PopulationGeneratorFactory
from hgp_lib.rules import ComplexityCheck

from .. import evaluation_artifacts
from ..data import DATASET_NAMES, N_SPLITS
from ..rule_artifacts import load_artifact, prepare_fold

# Every backend with the values of each of its options. The suite benchmarks every
# combination of the values; add a backend or an option value here to benchmark it.
BACKEND_OPTIONS = {
    NumpyBackend: {
        "order": ("F", "C"),
        "low_memory": (True, False),
        "batched": (False, True),
    },
}

# TorchBackend is benchmarked when PyTorch is installed, on the devices this machine has.
TORCH_DEVICES = ("cpu", "cuda", "mps")
try:
    import torch

    from hgp_lib.evaluation.torch import TorchBackend
except ImportError:
    pass
else:
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    BACKEND_OPTIONS[TorchBackend] = {
        "device": tuple(device for device in TORCH_DEVICES if available[device]),
        "batched": (False, True),
    }

TILES = 10
RANDOM_RULE_ROWS = (1_000, 10_000)
PREDICT_ROWS = 10_000
TRAINING_DATASETS = ("breast_cancer", "spambase")
TRAINING_GENERATIONS = 100
TRAINING_POPULATION_SIZE = 100
SEED = 4101

pytestmark = pytest.mark.benchmark(disable_gc=True, warmup=False)


def backend_label(backend_class: type, options: dict) -> str:
    arguments = ", ".join(f"{name}={value!r}" for name, value in options.items())
    return f"{backend_class.__name__}({arguments})"


def backend_parameters() -> list:
    parameters = []
    for backend_class, options in BACKEND_OPTIONS.items():
        names = list(options)
        for values in itertools.product(*(options[name] for name in names)):
            kwargs = dict(zip(names, values))
            label = backend_label(backend_class, kwargs)
            parameters.append(pytest.param(backend_class(**kwargs), id=label))
    return parameters


BACKENDS = backend_parameters()


def backend_options(backend) -> dict:
    """The backend's options, as JSON values (a ``torch.device`` becomes ``"cpu"``)."""
    return {
        name: value
        if isinstance(value, (bool, int, float, str, type(None)))
        else str(value)
        for name, value in asdict(backend).items()
    }


def record(benchmark, scenario_id: str, backend, result: float, **info) -> None:
    options = backend_options(backend)
    benchmark.extra_info.update(
        {
            "scenario_id": scenario_id,
            "backend": backend_label(type(backend), options),
            "backend_options": options,
            "is_default": backend == type(backend)(),
            "test_score": float(result),
            **info,
        }
    )


# Inputs are loaded once and shared by every backend combination, so binarizing and
# loading artifacts are not repeated for each case, and are never timed. Shared arrays are
# read-only, so a backend that writes into its input fails instead of changing the input
# of the next cases. Rules are copied for each case, for the same reason.
@lru_cache(maxsize=None)
def training_fold(dataset: str, fold: int) -> tuple[np.ndarray, np.ndarray]:
    train_data, train_labels, *_ = prepare_fold(dataset, fold)
    train_data.setflags(write=False)
    train_labels.setflags(write=False)
    return train_data, train_labels


@lru_cache(maxsize=None)
def _evolved_rules(dataset: str, fold: int) -> tuple:
    return tuple(load_artifact(dataset, fold)[0])


@lru_cache(maxsize=None)
def _random_rules(num_literals: int) -> tuple:
    return tuple(evaluation_artifacts.load_artifact(num_literals))


def evolved_rules(dataset: str, fold: int) -> list:
    return [rule.copy() for rule in _evolved_rules(dataset, fold)]


def random_rules(num_literals: int) -> list:
    return [rule.copy() for rule in _random_rules(num_literals)]


def random_data(num_rows: int, num_features: int, seed: int):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 2, size=(num_rows, num_features), dtype=np.int8).astype(bool)
    labels = rng.integers(0, 2, size=num_rows).astype(bool)
    data.setflags(write=False)
    return data, labels


def mean_score(scores) -> float:
    return float(np.mean([np.mean(fold_scores) for fold_scores in scores]))


# Scores the 100 evolved rules of each of the five folds on their training rows, merged
# into sample weights as during training.
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dataset", DATASET_NAMES)
def test_population_scoring(benchmark, dataset, backend):
    scorer = Scorer(fast_f1_score, merge_rows=True)
    folds = [
        (
            backend.bind(Dataset(*training_fold(dataset, fold)), scorer),
            evolved_rules(dataset, fold),
        )
        for fold in range(N_SPLITS)
    ]
    benchmark.group = f"population_scoring.{dataset}"

    scores = benchmark(lambda: [evaluator.score(rules) for evaluator, rules in folds])
    record(benchmark, benchmark.group, backend, mean_score(scores), dataset=dataset)


# The same rules on the training rows repeated ``TILES`` times, without merging, as for a
# larger dataset without duplicate rows.
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dataset", DATASET_NAMES)
def test_population_scoring_tiled(benchmark, dataset, backend):
    scorer = Scorer(fast_f1_score, merge_rows=False)
    folds = []
    for fold in range(N_SPLITS):
        data, labels = training_fold(dataset, fold)
        dataset_rows = Dataset(np.tile(data, (TILES, 1)), np.tile(labels, TILES))
        folds.append((backend.bind(dataset_rows, scorer), evolved_rules(dataset, fold)))
    benchmark.group = f"population_scoring.{dataset}.x{TILES}_rows"

    scores = benchmark.pedantic(
        lambda: [evaluator.score(rules) for evaluator, rules in folds],
        rounds=3,
        iterations=1,
    )
    record(benchmark, benchmark.group, backend, mean_score(scores), dataset=dataset)


# The 100 random rules of the ``evaluation_default`` artifacts, scored on random data.
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "num_rows", RANDOM_RULE_ROWS, ids=lambda rows: f"{rows:_}-rows"
)
@pytest.mark.parametrize(
    "num_literals",
    evaluation_artifacts.LITERAL_COUNTS,
    ids=lambda literals: f"{literals:_}-literals",
)
def test_random_rule_scoring(benchmark, num_literals, num_rows, backend):
    rules = random_rules(num_literals)
    data, labels = random_data(num_rows, num_literals, SEED + num_literals + num_rows)
    evaluator = backend.bind(
        Dataset(data, labels), Scorer(fast_f1_score, merge_rows=False)
    )
    benchmark.group = f"random_rule_scoring.{num_literals:_}_literals.{num_rows:_}_rows"

    scores = benchmark(evaluator.score, rules)
    record(
        benchmark,
        benchmark.group,
        backend,
        float(np.mean(scores)),
        num_literals=num_literals,
        num_rows=num_rows,
    )


# One-shot predictions on user data, as `predict` does: the data is used as given, in
# row-major order, so ``order`` and ``batched`` do not apply.
@pytest.mark.parametrize("backend", BACKENDS)
def test_predict(benchmark, backend):
    num_literals = evaluation_artifacts.LITERAL_COUNTS[0]
    rules = random_rules(num_literals)
    data, _ = random_data(PREDICT_ROWS, num_literals, SEED)
    benchmark.group = f"predict.{num_literals:_}_literals.{PREDICT_ROWS:_}_rows"

    predictions = benchmark(lambda: [backend.predict(rule, data) for rule in rules])
    result = float(np.mean([np.count_nonzero(p) for p in predictions]) / PREDICT_ROWS)
    record(benchmark, benchmark.group, backend, result, num_rows=PREDICT_ROWS)


# Trains a flat population for ``TRAINING_GENERATIONS`` generations on the first fold,
# end to end: merging, binding, crossover, mutation, scoring and selection.
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dataset", TRAINING_DATASETS)
def test_training(benchmark, dataset, backend):
    data, labels = training_fold(dataset, 0)

    def train():
        np.random.seed(SEED)
        random.seed(SEED)
        gp = BooleanGP(
            BooleanGPConfig(
                train_data=data,
                train_labels=labels,
                population_factory=PopulationGeneratorFactory(
                    population_size=TRAINING_POPULATION_SIZE
                ),
                check_valid=ComplexityCheck(100),
                backend=backend,
            )
        )
        for _ in range(TRAINING_GENERATIONS):
            gp.step()
        return gp.global_best_score

    benchmark.group = f"training.{dataset}.{TRAINING_GENERATIONS}_generations"
    best_score = benchmark.pedantic(train, rounds=3, iterations=1)
    record(benchmark, benchmark.group, backend, best_score, dataset=dataset)
