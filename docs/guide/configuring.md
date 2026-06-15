# Configuring HGP

The population, mutation, and crossover behavior is configured through factories passed to `BooleanGPConfig`.
A factory holds config-time parameters and builds the actual component at runtime, once the number of features is known from the data.
This page covers the built-in factories and the hierarchical GP settings.
To replace a component with custom behavior, see [Extending HGP](extending.md).

## Factories

The `PopulationGeneratorFactory` and `MutationExecutorFactory` defer construction until `num_features` is known.
The default factories cover the common case.
`PopulationGeneratorFactory` uses `RandomStrategy`, and `MutationExecutorFactory` uses the standard literal and operator mutations.

```python
from hgp_lib.crossover import CrossoverExecutorFactory
from hgp_lib.mutations import MutationExecutorFactory
from hgp_lib.populations import PopulationGeneratorFactory
from hgp_lib.selections import TournamentSelection

population_factory = PopulationGeneratorFactory(population_size=100)
mutation_factory = MutationExecutorFactory(mutation_p=0.1)
crossover_factory = CrossoverExecutorFactory(
    crossover_p=0.7,
    crossover_strategy="random",  # or "best"
)
selection = TournamentSelection(tournament_size=10, selection_p=0.4)
```

Pass these to `BooleanGPConfig` to assemble a configured run.

```python
from hgp_lib.configs import BooleanGPConfig
from hgp_lib.utils.validation import ComplexityCheck

gp_config = BooleanGPConfig(
    train_data=train_data.to_numpy(dtype=bool),
    train_labels=train_labels,
    score_fn=score_fn,
    population_factory=population_factory,
    mutation_factory=mutation_factory,
    crossover_factory=crossover_factory,
    selection=selection,
    check_valid=ComplexityCheck(100),
    regeneration=True,
    regeneration_patience=100,
)
```

## Hierarchical GP

Hierarchical GP adds child populations that evolve on sampled subsets of features, then combine into larger rules.
Enable it through `max_depth`, `num_child_populations`, and a feature sampling strategy in `BooleanGPConfig`.

```python
from hgp_lib.populations import FeatureSamplingStrategy

gp_config.max_depth = 1
gp_config.num_child_populations = 3
gp_config.sampling_strategy = FeatureSamplingStrategy(feature_fraction=0.33)
```
