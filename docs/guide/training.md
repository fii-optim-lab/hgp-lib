# Training

The library uses dataclass configs ([`BooleanGPConfig`](../api/configs.md#hgp_lib.configs.boolean_gp_config.BooleanGPConfig), [`TrainerConfig`](../api/configs.md#hgp_lib.configs.trainer_config.TrainerConfig), [`BenchmarkerConfig`](../api/configs.md#hgp_lib.configs.benchmarker_config.BenchmarkerConfig)) for all main components.
When you pass training data in a config, the number of features is derived from the data (`train_data.shape[1]`).
It is then passed to the configured [`PopulationGeneratorFactory`](../api/populations.md#hgp_lib.populations.populations_factory.PopulationGeneratorFactory) and [`MutationExecutorFactory`](../api/mutations.md#hgp_lib.mutations.mutation_factory.MutationExecutorFactory) at runtime.
You do not need to pass `num_literals` when using the default factories.

Data passed to a trainer must be binarized first.
See [Data Preparation](data-preparation.md).

## Simple training

This runs a training with default hyperparameters.
Use [`BooleanGPConfig`](../api/configs.md#hgp_lib.configs.boolean_gp_config.BooleanGPConfig) and [`TrainerConfig`](../api/configs.md#hgp_lib.configs.trainer_config.TrainerConfig) to configure the run.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from hgp_lib.preprocessing import StandardBinarizer
from hgp_lib.configs import BooleanGPConfig, TrainerConfig
from hgp_lib.trainers import GPTrainer
from hgp_lib.utils.metrics import fast_f1_score

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# Binarize once: fit on train, apply to validation (avoids leakage).
binarizer = StandardBinarizer(num_bins=5)
train_data = binarizer.fit_transform(X_train, y_train.to_numpy())
val_data = binarizer.transform(X_val)
train_labels = y_train.to_numpy()
val_labels = y_val.to_numpy()

gp_config = BooleanGPConfig(
    train_data=train_data.to_numpy(dtype=bool),
    train_labels=train_labels,
    score_fn=fast_f1_score,
)
config = TrainerConfig(
    gp_config=gp_config,
    num_epochs=1000,
    val_data=val_data.to_numpy(dtype=bool),
    val_labels=val_labels,
)
trainer = GPTrainer(config)
result = trainer.fit()  # Returns PopulationHistory
```

## Configured training

Build a [`BooleanGPConfig`](../api/configs.md#hgp_lib.configs.boolean_gp_config.BooleanGPConfig) with custom factories and components, then wrap it in a [`TrainerConfig`](../api/configs.md#hgp_lib.configs.trainer_config.TrainerConfig).
The trainer accepts only [`TrainerConfig`](../api/configs.md#hgp_lib.configs.trainer_config.TrainerConfig).
See [Configuring HGP](configuring.md) and [Extending HGP](extending.md) for how to build the factories and components used below.

This example reuses `train_data`, `val_data`, `train_labels`, and `val_labels` from the setup above.

```python
from hgp_lib.configs import BooleanGPConfig, TrainerConfig
from hgp_lib.trainers import GPTrainer
from hgp_lib.crossover import CrossoverExecutorFactory
from hgp_lib.mutations import MutationExecutorFactory
from hgp_lib.populations import PopulationGeneratorFactory
from hgp_lib.selections import TournamentSelection
from hgp_lib.utils.metrics import fast_f1_score

population_factory = PopulationGeneratorFactory(population_size=100)
mutation_factory = MutationExecutorFactory(mutation_p=0.1, operator_p=0.5)
crossover_factory = CrossoverExecutorFactory(crossover_p=0.7)
selection = TournamentSelection()

def check_valid(rule):  # keep rules compact
    return len(rule) <= 25

gp_config = BooleanGPConfig(
    train_data=train_data.to_numpy(dtype=bool),
    train_labels=train_labels,
    score_fn=fast_f1_score,
    population_factory=population_factory,
    mutation_factory=mutation_factory,
    crossover_factory=crossover_factory,
    selection=selection,
    check_valid=check_valid,
    regeneration=True,
    regeneration_patience=100,
)
config = TrainerConfig(
    gp_config=gp_config,
    num_epochs=1000,
    val_data=val_data.to_numpy(dtype=bool),
    val_labels=val_labels,
    val_every=100,
)
trainer = GPTrainer(config)
result = trainer.fit()  # Returns PopulationHistory
```

## From raw data to a readable rule

This example runs the full path on a single split.
It binarizes the data once, trains a rule with [`GPTrainer`](../api/trainers.md#hgp_lib.trainers.gp_trainer.GPTrainer), predicts on the test set, and prints the rule as a readable expression.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from hgp_lib.preprocessing import StandardBinarizer
from hgp_lib.configs import BooleanGPConfig, TrainerConfig
from hgp_lib.trainers import GPTrainer
from hgp_lib.utils.metrics import fast_f1_score

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

binarizer = StandardBinarizer(num_bins=5)
train_bin = binarizer.fit_transform(X_train, y_train.to_numpy())
test_bin = binarizer.transform(X_test)

gp = BooleanGPConfig(
    score_fn=fast_f1_score,
    train_data=train_bin.to_numpy(),
    train_labels=y_train.to_numpy(),
)
history = GPTrainer(TrainerConfig(gp_config=gp, num_epochs=1000)).fit()

rule = history.global_best_rule
predictions = rule.evaluate(test_bin.to_numpy())
print(rule.to_str(binarizer.get_feature_names_out()))
```

`binarizer.get_feature_names_out()` returns the binarized column names in order, so the printed rule reads as plain logic (each literal index is replaced by its column name).
To skip the manual binarization, [`BooleanRuleClassifier`](../api/trainers.md#hgp_lib.trainers.boolean_rule_classifier.BooleanRuleClassifier) does the same end-to-end path in a few lines (see [Getting Started](../getting-started.md)).

## Predicting with a fitted trainer

After `fit`, the trainer exposes a scikit-learn style [`predict`](../api/trainers.md#hgp_lib.trainers.gp_trainer.GPTrainer.predict).
It evaluates the best rule found during training, so a fitted [`GPTrainer`](../api/trainers.md#hgp_lib.trainers.gp_trainer.GPTrainer) can be used where an estimator is expected.
The input must already be binarized (a boolean array) with the same feature layout as the training data.
Continuing from the previous example (reusing `gp` and `test_bin`):

```python
trainer = GPTrainer(TrainerConfig(gp_config=gp, num_epochs=1000))
trainer.fit()

predictions = trainer.predict(test_bin.to_numpy())  # 1-D boolean array
```

This is equivalent to calling `history.global_best_rule.evaluate(test_bin.to_numpy())`, but keeps the estimator-style API.

For end-to-end examples on real datasets, see the [Experiments](../experiments/index.md) section.
