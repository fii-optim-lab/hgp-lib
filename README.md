# Hierarchical Genetic Programming Library

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
```

## Quickstart

Binarize the data, train a rule with `GPTrainer`, then use it to predict and print it as plain logic.

```python
from hgp_lib.preprocessing import StandardBinarizer
from hgp_lib.configs import BooleanGPConfig, TrainerConfig
from hgp_lib.trainers import GPTrainer
from hgp_lib.utils.metrics import fast_f1_score

binarizer = StandardBinarizer(num_bins=5)
train_bin = binarizer.fit_transform(train_data, train_labels)
test_bin = binarizer.transform(test_data)

gp = BooleanGPConfig(
    score_fn=fast_f1_score,
    train_data=train_bin.to_numpy(),
    train_labels=train_labels,
)
history = GPTrainer(TrainerConfig(gp_config=gp, num_epochs=1000)).fit()

rule = history.global_best_rule
predictions = rule.evaluate(test_bin.to_numpy())
column_names = dict(enumerate(train_bin.columns))
print(rule.to_str(column_names))
```

The `column_names` map turns literal indices back into the binarized column names, so the printed rule reads as plain logic.
The [Data Preparation](https://fii-optim-lab.github.io/hgp-lib/guide/data-preparation/) guide shows how to use `StandardBinarizer` without leaking data between splits.

## Benchmarking

`GPBenchmarker` runs multiple independent experiments and aggregates the results.
Each run takes a stratified train/test split, performs k-fold cross-validation on the training set, and evaluates the best rule on the held-out test set.
Runs execute in parallel by default.

The benchmarker binarizes data internally, per fold, so you pass a raw `pandas.DataFrame` and skip manual binarization.

```python
import numpy as np
import pandas as pd
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.benchmarkers import GPBenchmarker

data = pd.DataFrame(...)  # raw features (bool / categorical / numeric)
labels = np.array(...)    # 1-D target array

gp_config = BooleanGPConfig(score_fn=score_fn)
trainer_config = TrainerConfig(gp_config=gp_config, num_epochs=1000, val_every=100)
config = BenchmarkerConfig(
    data=data,
    labels=labels,
    trainer_config=trainer_config,
    num_runs=30,
    n_folds=5,
    test_size=0.2,
    n_jobs=-1,
)
result = GPBenchmarker(config).fit()

test_scores = result.test_scores
print(f"Test score: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")

# Human-readable best rule
print(result.best_rule.to_str(result.best_run.feature_names))
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
