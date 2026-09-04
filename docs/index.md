# HGP Library

A Python library for **explainable rule-based classification** via Hierarchical Genetic Programming (HGP).

HGP evolves human-readable boolean rules (e.g. `And(age < 50, Or(income >= 30k, employed))`) that classify data.
The rule is the whole model, so a trained classifier can be read and explained directly.
Internally it combines genetic programming with hierarchical population structures, crossover, mutation, and selection operators.

## Key features

- Evolve interpretable boolean rule trees from tabular data
- Read the trained model directly, no post-hoc explainer needed
- Hierarchical GP with configurable child populations and feature/instance sampling
- Built-in benchmarking with stratified k-fold CV and parallel execution
- Automatic binarization of numeric and categorical features
- Scorer optimization via data deduplication and sample weights
- Configurable mutations, crossover, and selection strategies
- Dataclass-based configuration for reproducibility

## A readable model

A trained run returns a rule you can print with feature names and read as plain logic.

```python
print(result.best_rule.to_str(result.best_run.feature_names))
# And(income >= 30k, Or(employed, ~student))
```

Read it as: predict the positive class when income is at least 30k, and the person is either employed or not a student.
The model is the explanation, so there is nothing else to consult to know why a prediction was made.
See [Interpretability](interpretability.md) for why this matters.

## Quick example

Runs as-is on the scikit-learn `breast_cancer` dataset.
[`BooleanRuleClassifier`](api/trainers.md#hgp_lib.trainers.boolean_rule_classifier.BooleanRuleClassifier)
binarizes the raw data, evolves a rule, and applies the same binarization when predicting.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from hgp_lib import BooleanRuleClassifier
from hgp_lib.configs import BooleanGPConfig, TrainerConfig

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=0
)

config = TrainerConfig(
    gp_config=BooleanGPConfig(), num_epochs=500, val_every=50
)
clf = BooleanRuleClassifier(config)
clf.fit(X_train, y_train, X_val, y_val)  # validation is binarized internally too

predictions = clf.predict(X_test)
print(clf.format_rule())
```

## How it works

The method is genetic programming.
A population of candidate rules is scored against the data, the best rules are selected, and crossover and mutation produce the next generation.
Over many epochs the population converges toward rules with high fitness.
Hierarchical GP extends this with child populations that evolve on sampled subsets of features, then combine into larger rules.
See [Theory](theory.md) for the full picture and a comparison with decision trees.

Boolean GP operates on boolean data, so numeric and categorical columns are binarized first.
A numeric feature becomes a set of boolean bins.
The [Data Preparation](guide/data-preparation.md) guide covers this.

The model is a single boolean rule, which is readable on its own.
See [Interpretability](interpretability.md) for why this matters.

## Navigation

- [Getting Started](getting-started.md): installation and a first run
- [Theory](theory.md): how the GP search works and why it beats greedy trees
- [Interpretability](interpretability.md): readable rules and explainable models
- [Data Preparation](guide/data-preparation.md): binarization and avoiding leakage
- [Training](guide/training.md): [`GPTrainer`](api/trainers.md#hgp_lib.trainers.gp_trainer.GPTrainer) and run configuration
- [Benchmarking](guide/benchmarking.md): aggregated runs and scorer optimization
- [Configuring HGP](guide/configuring.md): factories and hierarchical GP settings
- [Extending HGP](guide/extending.md): custom strategies, mutations, and low-level use
- [Rule Trees](guide/rule-trees.md): the rule data structure and its speed optimizations
- [Experiments](experiments/index.md): reproducing dataset experiments (PMLB, PaySim, AEAC)
- [API Reference](api/index.md): full module documentation
