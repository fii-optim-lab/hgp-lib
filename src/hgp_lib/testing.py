"""
Testing utilities for evaluation backends.

`check_backend` checks that a backend gives the same results as a reference backend
(`NumpyBackend` by default) on small generated datasets. Use it to verify a custom
`EvaluationBackend`, or a new option of an existing one. It only needs NumPy, and raises
an ``AssertionError`` that names the first check that failed.
"""

import pickle
import random
from collections.abc import Callable

import numpy as np
from numpy import ndarray

from .evaluation import (
    Dataset,
    EvaluationBackend,
    NumpyBackend,
    fast_accuracy_score,
    fast_f1_score,
)
from .rules import And, Literal, Or, Rule

__all__ = ["check_backend"]

_NUM_ROWS = 120
_NUM_FEATURES = 6
_NUM_RULES = 40
_GENERATIONS = 8


def _random_rule(rng: np.random.Generator, depth: int) -> Rule:
    if depth == 0 or rng.random() < 0.3:
        return Literal(
            value=int(rng.integers(_NUM_FEATURES)), negated=bool(rng.random() < 0.5)
        )
    operator = And if rng.random() < 0.5 else Or
    subrules = [_random_rule(rng, depth - 1) for _ in range(rng.integers(2, 5))]
    return operator(subrules, negated=bool(rng.random() < 0.5), copy_subrules=False)


def _rules(rng: np.random.Generator) -> list[Rule]:
    rules = [_random_rule(rng, 3) for _ in range(_NUM_RULES)]
    # Single literals, and operators whose children are all negated literals.
    rules += [Literal(value=0), Literal(value=1, negated=True)]
    rules.append(And([Literal(value=i, negated=True) for i in range(3)], negated=True))
    rules.append(Or([Literal(value=i, negated=True) for i in range(3, 6)]))
    return rules


def _datasets(rng: np.random.Generator) -> dict[str, Dataset]:
    """Small datasets with duplicate rows, covering the label edge cases."""
    base = rng.integers(0, 2, size=(30, _NUM_FEATURES)).astype(bool)
    rows = rng.integers(0, len(base), size=_NUM_ROWS)
    data = base[rows]
    labels = rng.integers(0, 2, size=len(base)).astype(bool)[rows]
    return {
        "mixed labels": Dataset(data, labels),
        "no positive labels": Dataset(data, np.zeros(_NUM_ROWS, dtype=bool)),
        "only positive labels": Dataset(data, np.ones(_NUM_ROWS, dtype=bool)),
    }


def _custom_f1(y_true, y_pred, sample_weight=None):
    """A custom scorer that only uses operations shared by NumPy arrays and tensors."""
    if sample_weight is None:
        tp = float((y_true & y_pred).sum())
        predicted = float(y_pred.sum())
        positives = float(y_true.sum())
    else:
        tp = float((sample_weight * (y_true & y_pred)).sum())
        predicted = float((sample_weight * y_pred).sum())
        positives = float((sample_weight * y_true).sum())
    if positives == 0:
        return 1.0 if predicted == 0 else 0.0
    return 2 * tp / (predicted + positives)


def _custom_accuracy(y_true, y_pred):
    """A custom scorer without ``sample_weight``."""
    return float((y_true == y_pred).sum()) / len(y_true)


class _BatchedF1:
    """A custom scorer with a ``batched`` method."""

    def __call__(self, y_true, y_pred, sample_weight=None):
        return _custom_f1(y_true, y_pred, sample_weight)

    def batched(self, y_true, y_preds, sample_weight=None):
        return [_custom_f1(y_true, y_pred, sample_weight) for y_pred in y_preds]


def _fail(check: str, detail: str = "") -> None:
    raise AssertionError(f"{check}: {detail}" if detail else check)


def _equal(check: str, actual, expected) -> None:
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as error:
        raise AssertionError(f"{check}: {error}") from None


def _scores(backend: EvaluationBackend, evaluator, rules: list[Rule]) -> ndarray:
    return backend.to_numpy(evaluator.score(rules))


def _check_predict(
    backend: EvaluationBackend, reference: EvaluationBackend, rng: np.random.Generator
) -> None:
    rules = _rules(rng)
    data = rng.integers(0, 2, size=(_NUM_ROWS, _NUM_FEATURES)).astype(bool)
    read_only = np.asfortranarray(data)
    read_only.setflags(write=False)
    for layout, values in (("row-major", data), ("read-only column-major", read_only)):
        original = values.copy()
        for rule in rules:
            expected = reference.to_numpy(
                reference.predict(rule, reference.from_numpy(values))
            )
            actual = backend.to_numpy(backend.predict(rule, backend.from_numpy(values)))
            _equal(f"predict on {layout} data, rule {rule}", actual, expected)
        _equal(f"predict must not change {layout} data", values, original)


def _check_scores(
    backend: EvaluationBackend, reference: EvaluationBackend, rng: np.random.Generator
) -> None:
    rules = _rules(rng)
    scorers: list[tuple[str, Callable, tuple[bool, ...]]] = [
        ("fast_f1_score", fast_f1_score, (False, True)),
        ("fast_accuracy_score", fast_accuracy_score, (False, True)),
        ("a custom scorer with sample_weight", _custom_f1, (False, True)),
        ("a custom scorer without sample_weight", _custom_accuracy, (False,)),
        ("a custom scorer with a batched method", _BatchedF1(), (False, True)),
    ]
    for name, dataset in _datasets(rng).items():
        for scorer_name, score_fn, merges in scorers:
            for merge_rows in merges:
                check = f"scores of {scorer_name} on {name}, merge_rows={merge_rows}"
                expected_evaluator = reference.bind(
                    dataset, score_fn, merge_rows=merge_rows
                )
                evaluator = backend.bind(dataset, score_fn, merge_rows=merge_rows)
                if evaluator.dataset.n_rows != dataset.n_rows:
                    _fail(check, "the bound dataset has a different number of rows")
                expected = _scores(reference, expected_evaluator, rules)
                actual = _scores(backend, evaluator, rules)
                if actual.dtype != expected.dtype:
                    _fail(check, f"dtype {actual.dtype}, expected {expected.dtype}")
                _equal(check, actual, expected)
                _equal(
                    f"{check}, no rules", _scores(backend, evaluator, []).shape, (0,)
                )
                for rule in rules[:10]:
                    counts = evaluator.confusion_matrix(rule)
                    if counts != expected_evaluator.confusion_matrix(rule):
                        _fail(f"confusion matrix on {name}", f"rule {rule}: {counts}")


def _native(backend: EvaluationBackend, dataset) -> Dataset:
    weights = dataset.sample_weight
    return Dataset(
        backend.to_numpy(dataset.data),
        backend.to_numpy(dataset.labels),
        None if weights is None else backend.to_numpy(weights),
    )


def _check_datasets(
    backend: EvaluationBackend, reference: EvaluationBackend, rng: np.random.Generator
) -> None:
    dataset = _datasets(rng)["mixed labels"]
    rules = _rules(rng)
    for merge_rows in (False, True):
        evaluator = backend.bind(dataset, merge_rows=merge_rows)
        expected_evaluator = reference.bind(dataset, merge_rows=merge_rows)
        native, expected = evaluator.dataset, expected_evaluator.dataset
        rows = np.sort(rng.choice(expected.n_rows, _NUM_ROWS // 2, replace=False))
        columns = np.array([4, 0, 2])
        pairs = {
            "the bound dataset": (native, expected),
            "take": (native.take(rows), expected.take(rows)),
            "select_features": (
                native.select_features(columns),
                expected.select_features(columns),
            ),
            "deduplicate after select_features": (
                native.select_features(columns).deduplicate(),
                expected.select_features(columns).deduplicate(),
            ),
        }
        for operation, (actual, wanted) in pairs.items():
            check = f"{operation}, merge_rows={merge_rows}"
            converted = _native(backend, actual)
            _equal(f"{check}: data", converted.data, wanted.data)
            _equal(f"{check}: labels", converted.labels, wanted.labels)
            if (converted.sample_weight is None) != (wanted.sample_weight is None):
                _fail(check, "sample_weight differs")
            if wanted.sample_weight is not None:
                _equal(
                    f"{check}: sample_weight",
                    converted.sample_weight,
                    wanted.sample_weight,
                )
            if (actual.n_rows, actual.n_features) != (wanted.n_rows, wanted.n_features):
                _fail(check, "n_rows or n_features differs")
        _equal(
            f"subset scores, merge_rows={merge_rows}",
            _scores(backend, evaluator.subset(rows), rules),
            _scores(reference, expected_evaluator.subset(rows), rules),
        )


def _check_pickle(backend: EvaluationBackend, rng: np.random.Generator) -> None:
    restored = pickle.loads(pickle.dumps(backend))
    if restored != backend:
        _fail("pickling", "the restored backend is not equal to the original")
    rules = _rules(rng)
    evaluator = backend.bind(_datasets(rng)["mixed labels"], merge_rows=True)
    restored_evaluator = pickle.loads(pickle.dumps(evaluator))
    _equal(
        "pickling an evaluator",
        _scores(backend, restored_evaluator, rules),
        _scores(backend, evaluator, rules),
    )


def _history(backend: EvaluationBackend, seed: int, options: dict) -> list:
    # Imported here: hgp_lib.algorithms imports the evaluation package.
    from .algorithms import BooleanGP
    from .configs import BooleanGPConfig
    from .populations import PopulationGeneratorFactory

    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    dataset = _datasets(rng)["mixed labels"]
    options = {
        "population_factory": PopulationGeneratorFactory(population_size=20),
        **options,
    }
    gp = BooleanGP(
        BooleanGPConfig(
            train_data=dataset.data,
            train_labels=dataset.labels,
            backend=backend,
            **options,
        )
    )
    history = []
    for _ in range(_GENERATIONS):
        metrics = gp.step()
        history.append((metrics.best_train_score, str(metrics.best_rule)))
    history.append(gp.evaluator.confusion_matrix(gp.global_best_rule))
    return history


def _check_training(
    backend: EvaluationBackend, reference: EvaluationBackend, seed: int
) -> None:
    from .populations import (
        BestLiteralStrategy,
        CombinedSamplingStrategy,
        FeatureSamplingStrategy,
        InstanceSamplingStrategy,
        PopulationGeneratorFactory,
    )

    class BestLiterals(PopulationGeneratorFactory):
        def create_strategies(self, num_literals, evaluator):
            return [BestLiteralStrategy(num_literals, evaluator, sample_size=0.5)]

    def hierarchy(strategy):
        return {
            "max_depth": 1,
            "num_child_populations": 2,
            "sampling_strategy": strategy,
            "top_k_transfer": 5,
        }

    runs = {
        "a flat population": {},
        "a custom scorer": {"score_fn": _custom_f1, "optimize_scorer": True},
        "BestLiteralStrategy with sample_size": {
            "population_factory": BestLiterals(population_size=20)
        },
        "feature sampling": hierarchy(FeatureSamplingStrategy(feature_fraction=0.7)),
        "instance sampling": hierarchy(InstanceSamplingStrategy(sample_fraction=0.5)),
        "combined sampling": hierarchy(
            CombinedSamplingStrategy(feature_fraction=0.7, sample_fraction=0.5)
        ),
    }
    for name, options in runs.items():
        actual = _history(backend, seed, options)
        expected = _history(reference, seed, options)
        if actual != expected:
            _fail(f"training with {name}", f"history {actual}, expected {expected}")


def check_backend(
    backend: EvaluationBackend,
    reference: EvaluationBackend | None = None,
    *,
    seed: int = 0,
) -> None:
    """
    Check that ``backend`` gives the same results as ``reference``.

    Checks, on small generated datasets with duplicate rows:

    - ``predict`` on random rules (row-major and read-only column-major data, which must
      not change).
    - Scores and confusion matrices for `fast_f1_score`, `fast_accuracy_score` and
      custom scorers (with and without ``sample_weight``, and with a ``batched``
      method), with and without merged rows, including labels that are all negative
      or all positive. Scores must be equal, with the same dtype.
    - The dataset operations ``take``, ``select_features`` and ``deduplicate`` of the
      evaluator's dataset, and ``Evaluator.subset``.
    - Pickling the backend and an evaluator.
    - Training with fixed seeds: a flat population, a custom scorer,
      ``BestLiteralStrategy`` with ``sample_size``, and hierarchical populations with
      each sampling strategy. The best scores and rules of every generation must be
      equal.

    Custom scorers are written with operations that NumPy arrays and PyTorch tensors
    share, so they also check that a backend passes its own arrays to them.

    Args:
        backend (EvaluationBackend): The backend to check.
        reference (EvaluationBackend | None): The backend to compare with. ``None``
            uses ``NumpyBackend`` with the score dtype of ``backend``. Default: `None`.
        seed (int): Seed for the generated data and rules. Default: `0`.

    Raises:
        AssertionError: If a result differs. The message names the check.

    Examples:
        >>> from hgp_lib.evaluation import NumpyBackend
        >>> from hgp_lib.testing import check_backend
        >>> check_backend(NumpyBackend(evaluation="block", batched=True, batch_size=7))
    """
    rng = np.random.default_rng(seed)
    if reference is None:
        probe = backend.bind(_datasets(rng)["mixed labels"])
        dtype = _scores(backend, probe, [Literal(value=0)]).dtype
        reference = NumpyBackend(dtype=dtype)
    _check_predict(backend, reference, np.random.default_rng(seed))
    _check_scores(backend, reference, np.random.default_rng(seed + 1))
    _check_datasets(backend, reference, np.random.default_rng(seed + 2))
    _check_pickle(backend, np.random.default_rng(seed + 3))
    _check_training(backend, reference, seed)
