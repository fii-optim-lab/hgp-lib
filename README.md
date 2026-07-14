# Hierarchical Genetic Programming Library

[![CI](https://github.com/fii-optim-lab/hgp-lib/actions/workflows/python-package.yml/badge.svg)](https://github.com/fii-optim-lab/hgp-lib/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/fii-optim-lab/hgp-lib/branch/master/graph/badge.svg)](https://codecov.io/gh/fii-optim-lab/hgp-lib)
[![PyPI version](https://img.shields.io/pypi/v/hgp-lib.svg)](https://pypi.org/project/hgp-lib/)
[![Python versions](https://img.shields.io/pypi/pyversions/hgp-lib.svg)](https://pypi.org/project/hgp-lib/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://fii-optim-lab.github.io/hgp-lib/)

A Python library for explainable rule-based classification.
It evolves human-readable boolean rule trees via hierarchical genetic programming, with automatic binarization and parallel benchmarking.

Full documentation: <https://fii-optim-lab.github.io/hgp-lib/>

## What it does

`hgp_lib` evolves boolean rules that classify tabular data.
A rule is a tree of logical operators (`And`, `Or`) over literals, for example `And(age < 50, Or(income >= 30k, employed))`.
Rules are readable, so a trained classifier can be inspected and explained.

The method is genetic programming.
A population of candidate rules is scored against the data, the best rules are selected, and crossover and mutation produce the next generation.
Over many epochs the population converges toward rules with high fitness.
Hierarchical GP extends this with child populations that evolve on sampled subsets of features, then combine into larger rules.

Boolean GP operates on boolean data.
Numeric and categorical columns are binarized first, so a numeric feature becomes a set of boolean bins.
See [Data Preparation](https://fii-optim-lab.github.io/hgp-lib/guide/data-preparation/) for details.

The model is a single boolean rule, so it is readable on its own and needs no separate explanation.
See [Theory](https://fii-optim-lab.github.io/hgp-lib/theory/) for how the search works and [Interpretability](https://fii-optim-lab.github.io/hgp-lib/interpretability/) for why this matters.

## Installation

```bash
pip install hgp-lib
# or
pip install 'hgp-lib[dev]'
```

## Quickstart

`BooleanRuleClassifier` is the fastest way to train an interpretable rule end to end.
It binarizes the raw data for you, evolves a rule, and applies the same binarization when predicting.
The example below is fully runnable on the scikit-learn `breast_cancer` dataset.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from hgp_lib import BooleanRuleClassifier
from hgp_lib.configs import BooleanGPConfig, TrainerConfig
from hgp_lib.utils.metrics import fast_f1_score

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

config = TrainerConfig(gp_config=BooleanGPConfig(score_fn=fast_f1_score), num_epochs=1000)
clf = BooleanRuleClassifier(config)  # StandardBinarizer by default; pass binarizer=... to customize
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)  # raw data is binarized internally
print(clf.format_rule())           # the evolved rule as plain logic
```

`clf.format_rule()` prints the rule with the binarized column names, so the model reads as plain logic.
To binarize and train manually with `GPTrainer`, see [Training](https://fii-optim-lab.github.io/hgp-lib/guide/training/); the [Data Preparation](https://fii-optim-lab.github.io/hgp-lib/guide/data-preparation/) guide shows how to avoid leaking data between splits.

## Benchmarking

`GPBenchmarker` runs multiple independent experiments and aggregates the results.
Each run takes a stratified train/test split, performs k-fold cross-validation on the training set, and evaluates the best rule on the held-out test set.
Runs execute in parallel by default.

The benchmarker binarizes data internally, per fold, so you pass a raw `pandas.DataFrame` and skip manual binarization.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.benchmarkers import GPBenchmarker
from hgp_lib.utils.metrics import fast_f1_score

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

gp_config = BooleanGPConfig(score_fn=fast_f1_score)
trainer_config = TrainerConfig(gp_config=gp_config, num_epochs=1000, val_every=100)
config = BenchmarkerConfig(
    data=X,
    labels=y.to_numpy(),
    trainer_config=trainer_config,
    num_runs=30,
    n_folds=5,
    test_size=0.2,
    n_jobs=-1,
)
benchmarker = GPBenchmarker(config)
result = benchmarker.fit()

test_scores = result.test_scores
print(f"Test score: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")

# Human-readable best rule
print(result.best_rule.to_str(result.best_run.feature_names))

# sklearn-style predict on raw data (binarized internally with the best run's binarizer)
predictions = benchmarker.predict(X)
```

See [Benchmarking](https://fii-optim-lab.github.io/hgp-lib/guide/benchmarking/) for scorer optimization, custom binarizers, and the aggregated result fields.

## Customizing the algorithm

The population, mutation, and crossover behavior is configured through factories passed to `BooleanGPConfig`.
The default factories cover the common case.
To use custom initialization strategies or mutations, subclass a factory and override its construction hook.

```python
from hgp_lib.populations import PopulationGeneratorFactory

factory = PopulationGeneratorFactory(population_size=100)
```

The [Configuring HGP](https://fii-optim-lab.github.io/hgp-lib/guide/configuring/) guide covers the built-in factories and hierarchical GP.
The [Extending HGP](https://fii-optim-lab.github.io/hgp-lib/guide/extending/) guide covers custom strategies, mutations, and low-level use of `BooleanGP` directly.

## Documentation

- [Getting Started](https://fii-optim-lab.github.io/hgp-lib/getting-started/)
- [Theory](https://fii-optim-lab.github.io/hgp-lib/theory/)
- [Interpretability](https://fii-optim-lab.github.io/hgp-lib/interpretability/)
- [Data Preparation](https://fii-optim-lab.github.io/hgp-lib/guide/data-preparation/)
- [Training](https://fii-optim-lab.github.io/hgp-lib/guide/training/)
- [Benchmarking](https://fii-optim-lab.github.io/hgp-lib/guide/benchmarking/)
- [Configuring HGP](https://fii-optim-lab.github.io/hgp-lib/guide/configuring/)
- [Extending HGP](https://fii-optim-lab.github.io/hgp-lib/guide/extending/)
- [Rule Trees](https://fii-optim-lab.github.io/hgp-lib/guide/rule-trees/)
- [Experiments](https://fii-optim-lab.github.io/hgp-lib/experiments/)
- [API Reference](https://fii-optim-lab.github.io/hgp-lib/api/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
