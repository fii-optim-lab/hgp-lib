from hgp_lib.utils.metrics import fast_f1_score

# Benchmarking

The benchmarker runs multiple full runs (default 30), each with a stratified train/test split and k-fold CV on the training set.
Results are aggregated across runs.
Runs execute in parallel by default.
The benchmarker accepts a [`BenchmarkerConfig`](../api/configs.md#hgp_lib.configs.benchmarker_config.BenchmarkerConfig) containing a [`TrainerConfig`](../api/configs.md#hgp_lib.configs.trainer_config.TrainerConfig) template.

## Automatic binarization

!!! note
    Pass raw (non-binarized) data as a `pandas.DataFrame` in `BenchmarkerConfig.data`.
    For each fold, a fresh copy of the binarizer is fit on the training fold (with labels for supervised binning) and used to transform the validation fold.
    The best fold's binarizer is then used to transform the held-out test set.
    This prevents data leakage across folds and between train/test splits.

By default a [`StandardBinarizer`](../api/preprocessing.md#hgp_lib.preprocessing.binarizer.StandardBinarizer) is used.
You can pass a custom binarizer (unfitted) via the `binarizer` parameter:

```python
from hgp_lib.preprocessing import StandardBinarizer

binarizer = StandardBinarizer(num_bins=10)  # must be unfitted
config = BenchmarkerConfig(
    data=data,
    labels=labels,
    trainer_config=trainer_config,
    binarizer=binarizer,  # None -> default StandardBinarizer(num_bins=5)
)
```

See [Binarization](binarization.md) for how the binarizer works and its parameters.

## Feature names

The [`RunResult`](../api/metrics.md#hgp_lib.metrics.results.RunResult) includes `feature_names`, a `list[str]` of the binarized column names in order, index-aligned so that `feature_names[i]` names the feature a literal references with index `i`.
Use this to display rules in human-readable form:

```python
best_run = result.best_run
print(result.best_rule.to_str(best_run.feature_names))
```

### What "best run" means

In a k-fold setup there is no single obvious best run, so the selection is defined in two steps.

Within a run, the best fold is the fold with the highest validation score.
The best rule of that run is the best rule found in that fold.

Across runs, the best run is the one with the highest mean validation score across its folds.
The best rule of the whole experiment is the best rule from the best fold of that best run.

!!! note
    The best run is chosen by validation score, not by test score.
    The test score is held out and only measures the selected rule, so it does not drive the selection.

## Scorer optimization

The benchmarker can optimize scorers per fold by deduplicating data and using sample weights.
This speeds up scoring for datasets with many duplicate rows.
To use it, pass a base scorer (not pre-optimized) that accepts a `sample_weight` parameter, and set `optimize_scorer=True` in [`BooleanGPConfig`](../api/configs.md#hgp_lib.configs.boolean_gp_config.BooleanGPConfig) (the default).

Do not pass pre-optimized scorers (e.g. from [`optimize_scorers_for_data`](../api/utils.md#hgp_lib.utils.metrics.optimize_scorers_for_data)) to the benchmarker.
Pre-optimized scorers have sample weights bound to the original data, which become invalid after train/test/fold splits.
Either pass a base scorer with `optimize_scorer=True` (default), or use `optimize_scorer=False` for scorers without `sample_weight` support.

A scorer that supports sample weights looks like this:

```python
import numpy as np

def f1_score(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        tp = (y_true & y_pred).sum()
        pred_sum, label_sum = y_pred.sum(), y_true.sum()
    else:
        tp = np.dot(y_pred & y_true, sample_weight)
        pred_sum = np.dot(y_pred, sample_weight)
        label_sum = np.dot(y_true, sample_weight)
    if pred_sum == 0 or label_sum == 0:
        return 1.0 if pred_sum == label_sum == 0 else 0.0
    return 2 * tp / (pred_sum + label_sum)
```

## Full example

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.utils.metrics import fast_f1_score

X, y = load_breast_cancer(return_X_y=True, as_frame=True)  # raw DataFrame + target

# Nested configs: BooleanGPConfig -> TrainerConfig -> BenchmarkerConfig.
# train_data/train_labels are not needed in gp_config here;
# the benchmarker binarizes and sets them per fold.
gp_config = BooleanGPConfig(
    score_fn=fast_f1_score,
)
trainer_config = TrainerConfig(
    gp_config=gp_config,
    num_epochs=1000,
    val_every=100,
)
config = BenchmarkerConfig(
    data=X,
    labels=y.to_numpy(),
    trainer_config=trainer_config,
    num_runs=30,
    test_size=0.2,
    n_folds=5,
    n_jobs=-1,
)
benchmarker = GPBenchmarker(config)
result = benchmarker.fit()

# Aggregated metrics
test_scores = result.test_scores
print(f"Test score: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")

# Human-readable best rule
print(result.best_rule.to_str(result.best_run.feature_names))
```

## Predicting on new data

After `fit`, the benchmarker exposes a scikit-learn style [`predict`](../api/benchmarkers.md#hgp_lib.benchmarkers.gp_benchmarker.GPBenchmarker.predict) that works on raw data.
It binarizes the input with the best run's fitted binarizer (the one from its best fold), then evaluates the best rule.
Pass a `pandas.DataFrame` with the same columns, order, and dtypes as the data used to fit the benchmarker.

```python
benchmarker = GPBenchmarker(config)
benchmarker.fit()

# Same schema as the fitted data; here we reuse X from the example above.
predictions = benchmarker.predict(X)  # 1-D boolean array
```

The fitted binarizer is stored on [`RunResult.binarizer`](../api/metrics.md#hgp_lib.metrics.results.RunResult), so `predict` reproduces the exact encoding used during the best run.

## Hyperparameter tuning

Hyperparameter tuning runs on top of the benchmarker.
Each Optuna trial samples a configuration, runs a full benchmark with it, and uses the aggregated score as the trial objective.
The `scripts/optuna_hypertuning.py` script drives this loop.
For end-to-end tuning examples on real datasets, see the [Experiments](../experiments/index.md) section.
