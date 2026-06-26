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

For end-to-end examples on real datasets, see the [Experiments](../experiments/index.md) section.
