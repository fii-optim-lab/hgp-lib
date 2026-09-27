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
from hgp_lib.configs import BenchmarkerConfig
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

The [`RunResult`](../api/results.md#hgp_lib.results.experiment.RunResult) includes `feature_names`, a `list[str]` of the binarized column names in order, index-aligned so that `feature_names[i]` names the feature a literal references with index `i`.
Use this to display rules in human-readable form:

```python
result = GPBenchmarker(config).fit()
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

Training and validation rows with the same features and label can be merged into one row with an integer sample weight.
Rules are then evaluated on fewer rows, with exactly the same scores.
This happens separately for every fold, and for every child population in hierarchical GP.
The held-out test set is scored once, on its rows as they are.

`optimize_scorer` in [`BooleanGPConfig`](../api/configs.md#hgp_lib.configs.boolean_gp_config.BooleanGPConfig) decides when rows may be merged:

- `None` (default): for the built-in scorers ([`fast_f1_score`](../api/evaluation.md#hgp_lib.evaluation.scorers.fast_f1_score), [`fast_accuracy_score`](../api/evaluation.md#hgp_lib.evaluation.scorers.fast_accuracy_score)), never for a custom `score_fn`.
- `True`: also for a custom `score_fn` that accepts `sample_weight`. If it does not, a `FutureWarning` is emitted and rows are not merged.
- `False`: never.

A custom scorer that supports sample weights must give the same score for a row with weight `k` as for `k` copies of that row, like most scikit-learn metrics:

```python
from sklearn.metrics import balanced_accuracy_score

gp_config = BooleanGPConfig(score_fn=balanced_accuracy_score, optimize_scorer=True)
```

## Full example

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.benchmarkers import GPBenchmarker

X, y = load_breast_cancer(return_X_y=True, as_frame=True)  # raw DataFrame + target

# Nested configs: BooleanGPConfig -> TrainerConfig -> BenchmarkerConfig.
# train_data/train_labels are not needed in gp_config here;
# the benchmarker binarizes and sets them per fold.
gp_config = BooleanGPConfig()
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

# With the default settings, only the data is needed:
# GPBenchmarker(BenchmarkerConfig(data=X, labels=y.to_numpy())).fit()

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

The fitted binarizer is stored on [`RunResult.binarizer`](../api/results.md#hgp_lib.results.experiment.RunResult), so `predict` reproduces the exact encoding used during the best run.

## Hyperparameter tuning

Hyperparameter tuning runs on top of the benchmarker.
Each Optuna trial samples a configuration, runs a full benchmark with it, and uses the aggregated score as the trial objective.
The `scripts/optuna_hypertuning.py` script drives this loop.
For end-to-end tuning examples on real datasets, see the [Experiments](../experiments/index.md) section.
