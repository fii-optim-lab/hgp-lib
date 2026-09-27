# Changelog

## 2.0.0 (unreleased)

Rules are now evaluated by evaluation backends instead of by the rule classes, so rules only describe the logical structure.
This release has breaking changes; see the [migration guide](docs/migration.md).
Deprecated APIs keep working with a warning and will be removed in a future 2.x release.

### New Features

- New `hgp_lib.evaluation` package:
  - `predict(rule, data)` and `score(rule, data, labels, score_fn=None)` evaluate a rule on binarized data. They raise a `TypeError` for DataFrames and non-boolean arrays; use `BooleanRuleClassifier` or `GPBenchmarker` for raw data.
  - `fast_f1_score`, `fast_accuracy_score` and `confusion_matrix` score boolean predictions, with optional sample weights.
  - `NumpyBackend(order="F", low_memory=None, batched=False, batch_size=None)`, the default backend. Its options change speed and memory use, never results.
  - `TorchBackend(device=None, batched=False, batch_size=None)` evaluates rules with PyTorch on the CPU (the default) or a GPU, with the same scores as `NumpyBackend`. It needs the new `torch` extra: `pip install "hgp-lib[torch]"`.
  - `EvaluationBackend`, `Evaluator`, `Dataset` and `Scorer` for custom backends and strategies.
- `BooleanGPConfig.backend` selects the backend used for training, validation and prediction.
- Configs have defaults, so only the data is needed: `TrainerConfig()`, `BenchmarkerConfig(data=X, labels=y)` and `BooleanRuleClassifier()`. `TrainerConfig.num_epochs` defaults to `1000`.
- `hgp_lib.results` holds what training and benchmarking return (`GenerationMetrics`, `PopulationHistory`, `RunResult`, `ExperimentResult`).
- `ComplexityCheck` is available from `hgp_lib.rules`.
- `benchmarks/benchmark_backends.py` benchmarks every option combination of the evaluation backends on one version, and writes a report ranking them.

### API Changes

- `BooleanGPConfig.optimize_scorer` defaults to `None`: duplicate rows are merged for the built-in scorers only. Pass `optimize_scorer=True` to merge them for a custom scorer that accepts `sample_weight`.
- `PopulationGeneratorFactory.create` and `create_strategies` take `(num_literals, evaluator)`.
- `BestLiteralStrategy` takes `(num_literals, evaluator, sample_size=None, feature_size=None)`.
- `SamplingStrategy.sample` takes `(dataset, num_children)` and returns `SamplingResult(dataset, feature_mapping)`.
- `BooleanGP` has `backend`, `scorer` and `evaluator` attributes; `train_data` and `train_labels` are read-only views of the scored rows. Child populations receive their rows through the `dataset` argument.
- `GPTrainer` has a `val_evaluator` attribute.
- Evaluation backends support `Literal`, `And` and `Or`; other `Rule` subclasses raise a `TypeError`.
- Removed `hgp_lib.rules.low_memory_operators`; use `NumpyBackend(low_memory=True)`.
- Removed `SampleWeightScorer`, `optimize_scorers_for_data` and `transform_duplicates_to_sample_weight`; rows are merged by `EvaluationBackend.bind` (see `Dataset.deduplicate` and `Dataset.take`).
- Removed `BooleanGP.score_fn`, `BooleanGP.train_cm`, `BooleanGP.original_score_fn`, `GPTrainer.score_fn`, `GPTrainer.val_score_fn`, `GPTrainer.val_cm`, `GPTrainer.val_data` and `GPTrainer.val_labels`.

### Deprecations

- `Rule.evaluate(data)`: use `hgp_lib.evaluation.predict(rule, data)`.
- `hgp_lib.metrics`: renamed to `hgp_lib.results`.
- `hgp_lib.utils.metrics`: use `hgp_lib.evaluation`.
- `hgp_lib.utils.ComplexityCheck` and `hgp_lib.utils.validation.ComplexityCheck`: use `hgp_lib.rules.ComplexityCheck`.
- The `HGP_LOW_MEMORY` environment variable (`FutureWarning`): use `NumpyBackend(low_memory=...)`.
- `optimize_scorer=True` with a scorer that does not accept `sample_weight` emits a `FutureWarning`; it will raise a `ValueError` in a future release.

### Bug Fixes

- Child populations now keep the sample weights of their parent.
- `BestLiteralStrategy` no longer fails when `sample_size` is set and the data has duplicate rows.
- `BooleanGP` and the benchmarker no longer write `fast_f1_score` into the user's config when `score_fn` is `None`.

### Performance Improvements

- Training data is stored in column-major order, which speeds up rule evaluation.
- The scorer is bound to the training and validation rows once: the built-in scorers use dedicated kernels with the per-dataset constants computed at bind time, and the confusion matrix is computed from the same counts.
- The low-memory evaluation algorithm folds literal children into the result without temporary arrays, and is now the default. Scoring a population was 1.3x to 2.2x faster than with the previous default algorithm in the repository's backend benchmarks.
- `BestLiteralStrategy` scores literals on all rows through the bound kernels, on column-major data.
- The held-out test set is no longer deduplicated to score one rule once.

---

## [1.2.2](https://github.com/fii-optim-lab/hgp-lib/releases/tag/1.2.2)

Added separate module for benchmarking performance.

### API Changes
- Added `serialize` and `deserialize` methods for serializing and deserializing rules.

## [1.2.1](https://github.com/fii-optim-lab/hgp-lib/releases/tag/1.2.1)

### API Changes

- `BooleanGPConfig.score_fn` is now optional and defaults to `fast_f1_score`.

### Performance Improvements

- Scorer optimization is skipped when the input data is unique.


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
