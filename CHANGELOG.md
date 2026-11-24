# Changelog

## 2.0.0 (unreleased)

Rules are now evaluated by evaluation backends instead of by the rule classes, so rules only describe the logical structure.
This release has breaking changes; see the [migration guide](docs/migration.md).
Deprecated APIs keep working with a warning and will be removed in a future 2.x release.

### New Features

- New `hgp_lib.evaluation` package:
  - `predict(rule, data)` and `score(rule, data, labels, score_fn=fast_f1_score)` evaluate a rule on binarized data. They raise a `TypeError` for DataFrames and non-boolean arrays; use `BooleanRuleClassifier` or `GPBenchmarker` for raw data.
  - `fast_f1_score`, `fast_accuracy_score` and `confusion_matrix` score boolean predictions, with optional sample weights.
  - `NumpyBackend(order="F", evaluation="fused", batched=False, batch_size=None, dtype=float32)`, the default backend. `evaluation` selects one of three implementations (`"block"`, `"inplace"`, `"fused"`).
  - `TorchBackend(device=None, evaluation="fused", batched=False, batch_size=None, dtype=float32)` evaluates rules with PyTorch on the CPU (the default) or the given device. The rows, predictions and scores stay on the device. It needs PyTorch 2.0 or later.
  - `EvaluationBackend`, `Evaluator` and `Dataset` (`TorchDataset` for PyTorch) for custom backends and strategies. `Evaluator.subset` scores a subset of the original rows.
  - Scoring functions with a `batched` method are batched like the built-in scorers.
- `hgp_lib.testing.check_backend` checks that a backend gives the same results as the NumPy backend. `import hgp_lib` imports `hgp_lib.testing`, so it is reachable as an attribute.
- `BooleanGPConfig.backend` selects the backend used for training, validation and prediction.
- Configs have defaults, so only the data is needed: `TrainerConfig()`, `BenchmarkerConfig(data=X, labels=y)` and `BooleanRuleClassifier()`. `TrainerConfig.num_epochs` defaults to `1000`.
- `hgp_lib.results` holds what training and benchmarking return (`GenerationMetrics`, `PopulationHistory`, `RunResult`, `ExperimentResult`).
- `ComplexityCheck` is available from `hgp_lib.rules`.
- `benchmarks/benchmark_backends.py` benchmarks every option combination of the evaluation backends on one version, and writes a report ranking them.

### API Changes

- Scores are `float32` by default (backend option `dtype`). Weighted counts switch to `float64` when `float32` cannot count them exactly.
- Binarized data must be a boolean NumPy array; integer data raises a `TypeError`. Labels can be booleans or integers 0 and 1; float labels raise a `TypeError`, other integer values a `ValueError`.
- `BooleanGPConfig.score_fn` defaults to `fast_f1_score`; `score_fn=None` raises a `TypeError`.
- Scoring functions must return a finite float, never NaN. This is not checked.
- `BooleanGPConfig.optimize_scorer` defaults to `None`: duplicate rows are merged for the built-in scorers only. `optimize_scorer=True` merges them for a custom scorer too, which must then accept `sample_weight` (a `ValueError` otherwise).
- `PopulationGeneratorFactory.create` and `create_strategies` take `(num_literals, evaluator)`.
- `BestLiteralStrategy` takes `(num_literals, evaluator, sample_size=None, feature_size=None)`.
- `SamplingStrategy.sample` takes `(dataset, num_children)` and returns `SamplingResult(dataset, feature_mapping)`.
- `BooleanGP` has `backend` and `evaluator` attributes (`evaluator.score_fn` is the scoring function); `train_data` and `train_labels` are read-only views of the scored rows, in the backend's array type. Child populations receive their rows through the `dataset` argument.
- `GPTrainer` has a `val_evaluator` attribute.
- `Rule` is no longer an abstract base class.
- Evaluation backends support `Literal`, `And` and `Or`; other `Rule` subclasses raise a `TypeError`.
- Removed `hgp_lib.rules.low_memory_operators`; use `NumpyBackend(evaluation="inplace")`.
- Removed `SampleWeightScorer`, `optimize_scorers_for_data` and `transform_duplicates_to_sample_weight`; rows are merged by `EvaluationBackend.bind` (see `Dataset.deduplicate` and `Dataset.take`).
- Removed `BooleanGP.score_fn`, `BooleanGP.train_cm`, `BooleanGP.original_score_fn`, `GPTrainer.score_fn`, `GPTrainer.val_score_fn`, `GPTrainer.val_cm`, `GPTrainer.val_data` and `GPTrainer.val_labels`.

### Deprecations

- `Rule.evaluate(data)`: use `hgp_lib.evaluation.predict(rule, data)`.
- `hgp_lib.metrics`: renamed to `hgp_lib.results`.
- `hgp_lib.utils.metrics`: use `hgp_lib.evaluation`.
- `hgp_lib.utils.ComplexityCheck` and `hgp_lib.utils.validation.ComplexityCheck`: use `hgp_lib.rules.ComplexityCheck`.
- The `HGP_LOW_MEMORY` environment variable is ignored and emits a `FutureWarning`: use `NumpyBackend(evaluation=...)`.

### Bug Fixes

- Child populations now keep the sample weights of their parent.
- `BestLiteralStrategy` no longer fails when `sample_size` is set and the data has duplicate rows.
- `BooleanGP` and the benchmarker no longer write `fast_f1_score` into the user's config.
- Float labels are rejected when the configuration is validated, instead of failing after training. `BooleanRuleClassifier.fit` checks the labels before it fits the binarizer, and a binarizer that returns non-boolean columns fails at `fit`.
- A library warning, such as a deprecation, is no longer lost when it was first raised while the warning filters ignored it. Each message is still shown once per process, also under the `"always"` filter.
- The rule-evaluation benchmark artifacts (`benchmarks/artifacts/rule_evaluation`) are regenerated.

### Performance Improvements

- The scoring function is bound to the training and validation rows once: the built-in scorers use dedicated kernels with the per-dataset constants computed at bind time, and the confusion matrix is computed from the same counts.
- Population and sampling strategies work on the evaluator's dataset, so the training rows are stored once per population, in the backend's format.
- `BestLiteralStrategy` scores literals through the bound kernels.
- The held-out test set is no longer deduplicated to score one rule once.

---

## [1.2.4](https://github.com/fii-optim-lab/hgp-lib/releases/tag/1.2.4)

---

### API Changes

- Added new parameter to `StandardBinarizer`: `round_edges`, (default `True`), which rounds each numeric edge to the fewest decimals (at least 2) that keep every training value in the same bin.
- Numeric bin names no longer use a fixed number of decimals. For example, `x < 2.500` is now `x < 2.5`.
- Deprecated the `precision` parameter and the `column_precision` attribute of `StandardBinarizer`.


### Bug Fixes
- `StandardBinarizer` numeric bins are now closed on the left (`left <= x < right`). Previously, a value equal to an edge fell in the bin below it.
- `MutationExecutor` no longer returns a rule that fails `check_valid` when a rule receives several mutations and a later one is invalid.

---

## [1.2.3](https://github.com/fii-optim-lab/hgp-lib/releases/tag/1.2.3)

---

### Performance Improvements

- `BooleanGP` stores the training data in column-major order, which speeds up rule evaluation.

---

## [1.2.2](https://github.com/fii-optim-lab/hgp-lib/releases/tag/1.2.2)

Added separate module for benchmarking performance.

---

### API Changes
- Added `serialize` and `deserialize` methods for serializing and deserializing rules.

## [1.2.1](https://github.com/fii-optim-lab/hgp-lib/releases/tag/1.2.1)

### API Changes

- `BooleanGPConfig.score_fn` is now optional and defaults to `fast_f1_score`.

### Performance Improvements

- Scorer optimization is skipped when the input data is unique.

---

## [1.2.0](https://github.com/fii-optim-lab/hgp-lib/releases/tag/1.2.0)


### API Changes

- Methods called before fitting now raise `sklearn.exceptions.NotFittedError` instead of `RuntimeError` or `ValueError`.
- Binarizer schema mismatches now raise the dedicated `SchemaMismatchError`.
- `load_data` now raises `KeyError` when the target column is missing.
- Removed the unused `feature_indices` and `instance_indices` attributes from `SamplingResult`.

---

## [1.1.2](https://github.com/fii-optim-lab/hgp-lib/releases/tag/1.1.2)

No user-facing changes.

---

## [1.1.1](https://github.com/fii-optim-lab/hgp-lib/releases/tag/1.1.1)

### Bug Fixes

- Fixed incorrect behavior when calling `PopulationHistory.__len__`.

---

## [1.1.0](https://github.com/fii-optim-lab/hgp-lib/releases/tag/1.1.0)

### API Changes

- Added `BooleanRuleClassifier`, which combines binarization and Boolean genetic programming training into a scikit-learn-style classifier.
- Scoring functions now follow the scikit-learn argument order: `score_fn(y_true, y_pred)`.
- `Rule.to_str` now accepts feature names as a list instead of a dictionary.
- Added `get_feature_names_out` to binarizers.
- Custom `Binarizer` implementations must now implement `get_feature_names_out`.

---

## [1.0.1](https://github.com/fii-optim-lab/hgp-lib/releases/tag/1.0.1)

### API Changes

- Removed `optuna`, `optuna-dashboard`, and `matplotlib` from the runtime dependencies. These packages are now development dependencies.

---

## [1.0.0](https://github.com/fii-optim-lab/hgp-lib/releases/tag/1.0.0)

### API Changes

- Added support for missing values, string columns, and object columns to `StandardBinarizer`.
- Introduced an extensible binarization API for implementing custom binarizers.
- Added scikit-learn-style `predict` methods to `GPTrainer` and `GPBenchmarker`.

---

## [0.0.1](https://github.com/fii-optim-lab/hgp-lib/releases/tag/0.0.1)

Initial public release.
