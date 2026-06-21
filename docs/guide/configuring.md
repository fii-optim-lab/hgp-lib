# Configuring HGP

The population, mutation, and crossover behavior is configured through factories passed to [`BooleanGPConfig`](../api/configs.md#hgp_lib.configs.boolean_gp_config.BooleanGPConfig).
A factory holds config-time parameters and builds the actual component at runtime, once the number of features is known from the data.
This page covers the built-in factories and the hierarchical GP settings.
To replace a component with custom behavior, see [Extending HGP](extending.md).

## Factories

The [`PopulationGeneratorFactory`](../api/populations.md#hgp_lib.populations.populations_factory.PopulationGeneratorFactory) and [`MutationExecutorFactory`](../api/mutations.md#hgp_lib.mutations.mutation_factory.MutationExecutorFactory) defer construction until `num_features` is known.
The default factories cover the common case.
[`PopulationGeneratorFactory`](../api/populations.md#hgp_lib.populations.populations_factory.PopulationGeneratorFactory) uses [`RandomStrategy`](../api/populations.md#hgp_lib.populations.strategies.RandomStrategy), and [`MutationExecutorFactory`](../api/mutations.md#hgp_lib.mutations.mutation_factory.MutationExecutorFactory) uses the standard literal and operator mutations.

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

Pass these to [`BooleanGPConfig`](../api/configs.md#hgp_lib.configs.boolean_gp_config.BooleanGPConfig) to assemble a configured run.

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
    check_valid=ComplexityCheck(100),  # caps the rule at 100 nodes
    regeneration=True,
    regeneration_patience=100,
)
```

### Validity checks

`check_valid` is a predicate that takes a rule and returns `True` if the rule is allowed.
It is called after each mutation and crossover, and any rule it rejects is discarded.
The operation that produced it is retried, so `check_valid` shapes which rules the search is allowed to keep.

[`ComplexityCheck`](../api/utils.md#hgp_lib.utils.validation.ComplexityCheck) is the built-in case.
It rejects any rule larger than a node count, which caps the size of the evolved rules.

You can pass any callable with the same signature to enforce stricter constraints.
For example, a check that rejects redundant rules, where the same literal appears more than once, or one that rejects tautologies, where a rule is always true or always false.

```python
from hgp_lib.rules import Rule

def no_repeated_literals(rule: Rule) -> bool:
    values = [node.value for node in rule.flatten() if node.value is not None]
    return len(values) == len(set(values))
```

The check receives valid rules and decides which ones to keep, so it is the place to encode preferences the GP operators do not enforce on their own.

## Hierarchical GP

Hierarchical GP adds child populations that evolve on sampled subsets of features, then combine into larger rules.
Enable it through `max_depth`, `num_child_populations`, and a feature sampling strategy in [`BooleanGPConfig`](../api/configs.md#hgp_lib.configs.boolean_gp_config.BooleanGPConfig).

```python
from hgp_lib.populations import FeatureSamplingStrategy

gp_config.max_depth = 1
gp_config.num_child_populations = 3
gp_config.sampling_strategy = FeatureSamplingStrategy(feature_fraction=0.33)
```
