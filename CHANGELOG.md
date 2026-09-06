# Changelog

## Unreleased

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
