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
from hgp_lib.configs import BooleanGPConfig, TrainerConfig
from hgp_lib.trainers import GPTrainer

score_fn = ...  # scoring function: (predictions, labels) -> float
num_epochs = 1000

gp_config = BooleanGPConfig(
    train_data=train_data.to_numpy(dtype=bool),
    train_labels=train_labels,
    score_fn=score_fn,
)
config = TrainerConfig(
    gp_config=gp_config,
    num_epochs=num_epochs,
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

```python
from hgp_lib.configs import BooleanGPConfig, TrainerConfig
from hgp_lib.trainers import GPTrainer

gp_config = BooleanGPConfig(
    train_data=train_data.to_numpy(dtype=bool),
    train_labels=train_labels,
    score_fn=score_fn,
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
    num_epochs=num_epochs,
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

The `column_names` map turns the literal indices back into the binarized column names, so the printed rule reads as plain logic.

## Predicting with a fitted trainer

After `fit`, the trainer exposes a scikit-learn style [`predict`](../api/trainers.md#hgp_lib.trainers.gp_trainer.GPTrainer.predict).
It evaluates the best rule found during training, so a fitted [`GPTrainer`](../api/trainers.md#hgp_lib.trainers.gp_trainer.GPTrainer) can be used where an estimator is expected.
The input must already be binarized (a boolean array) with the same feature layout as the training data.

```python
trainer = GPTrainer(TrainerConfig(gp_config=gp, num_epochs=1000))
trainer.fit()

predictions = trainer.predict(test_bin.to_numpy())  # 1-D boolean array
```

This is equivalent to calling `history.global_best_rule.evaluate(test_bin.to_numpy())`, but keeps the estimator-style API.

For end-to-end examples on real datasets, see the [Experiments](../experiments/index.md) section.
