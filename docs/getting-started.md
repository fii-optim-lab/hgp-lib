# Getting Started

## Installation

```bash
pip install hgp-lib
# or
pip install 'hgp-lib[dev]'
```

## A first run

The fastest way to train an interpretable model is
[`BooleanRuleClassifier`](api/trainers.md#hgp_lib.trainers.boolean_rule_classifier.BooleanRuleClassifier).
It binarizes a raw `pandas.DataFrame` for you, evolves a rule, and applies the same
binarization when predicting. The example runs as-is on the scikit-learn
`breast_cancer` dataset.

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
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=0
)

config = TrainerConfig(
    gp_config=BooleanGPConfig(score_fn=fast_f1_score), num_epochs=1000, val_every=100
)
clf = BooleanRuleClassifier(config)  # StandardBinarizer by default; pass binarizer=... to customize
clf.fit(X_train, y_train, X_val, y_val)  # validation data is binarized internally too

predictions = clf.predict(X_test)  # raw data is binarized internally
print(clf.format_rule())           # the evolved rule as plain logic
```

Validation data is optional; when supplied it is binarized with the same fitted binarizer and used to track a validation score during training.

For a rigorous estimate over multiple runs and folds, use
[`GPBenchmarker`](api/benchmarkers.md#hgp_lib.benchmarkers.gp_benchmarker.GPBenchmarker),
which handles binarization, splitting, and aggregation for you:

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
    n_jobs=-1,
)
result = GPBenchmarker(config).fit()
print(f"Mean test F1: {np.mean(result.test_scores):.3f}")
print(result.best_rule.to_str(result.best_run.feature_names))
```

## Where to go next

- [Theory](theory.md): how the GP search works and why it beats greedy trees
- [Interpretability](interpretability.md): readable rules and explainable models
- [Data Preparation](guide/data-preparation.md): binarization and avoiding leakage
- [Training](guide/training.md): [`GPTrainer`](api/trainers.md#hgp_lib.trainers.gp_trainer.GPTrainer) and run configuration
- [Benchmarking](guide/benchmarking.md): aggregated runs and scorer optimization
- [Configuring HGP](guide/configuring.md): factories and hierarchical GP settings
- [Extending HGP](guide/extending.md): custom strategies, mutations, and low-level use
- [Rule Trees](guide/rule-trees.md): the rule data structure and its speed optimizations
- [Experiments](experiments/index.md): reproducing dataset experiments
- [API Reference](api/index.md): full module documentation
