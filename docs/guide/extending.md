# Extending HGP

The built-in factories cover the common case.
To change how rules are initialized or mutated, subclass a factory and override its construction hook.
For the built-in factories and hierarchical settings, see [Configuring HGP](configuring.md).

## Custom population strategies

The `PopulationGenerator` creates the initial set of rules.
It uses a strategy pattern to allow different initialization methods.

When using `BooleanGPConfig`, pass a `PopulationGeneratorFactory` rather than a `PopulationGenerator` directly.
Override `create_strategies` to use custom strategies.

```python
from hgp_lib.populations import (
    PopulationGeneratorFactory,
    RandomStrategy,
    BestLiteralStrategy,
)

class MyFactory(PopulationGeneratorFactory):
    def create_strategies(self, num_literals, score_fn, train_data, train_labels):
        random = RandomStrategy(num_literals=num_literals)
        best = BestLiteralStrategy(
            num_literals=num_literals,
            score_fn=score_fn,
            train_data=train_data,
            train_labels=train_labels,
            sample_size=100,
            feature_size=None,
        )
        return [random, best]

factory = MyFactory(population_size=100)
```

You can also create a `PopulationGenerator` directly for standalone use, outside of `BooleanGPConfig`.

```python
from hgp_lib.populations import PopulationGenerator, RandomStrategy

random_strategy = RandomStrategy(num_literals=10)
generator = PopulationGenerator(
    strategies=[random_strategy],
    population_size=100,
)
initial_population = generator.generate()
```

## Low-level use of BooleanGP

For full control over the training loop, use `BooleanGP` directly.
Training data is passed in the config, and `num_features` is derived from the data shape.
The number of features is then passed to the configured factories for runtime construction.

```python
from hgp_lib.configs import BooleanGPConfig
from hgp_lib.algorithms import BooleanGP
from hgp_lib.utils.validation import ComplexityCheck

check_valid = ComplexityCheck(100)

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
gp_algo = BooleanGP(gp_config)

for i in range(num_epochs):
    gen_metrics = gp_algo.step()
    if i % 100 == 0:
        val_score = gp_algo.evaluate_best(val_data.to_numpy(dtype=bool), val_labels)
        print(f"Epoch {i} -> val_best: {val_score:.4f}")

test_score = gp_algo.evaluate_best(test_data.to_numpy(dtype=bool), test_labels)
print(f"Test result: {test_score:.4f}")
```
