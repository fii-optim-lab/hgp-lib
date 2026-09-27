# Extending HGP

The built-in factories cover the common case.
To change how rules are initialized or mutated, subclass a factory and override its construction hook.
For the built-in factories and hierarchical settings, see [Configuring HGP](configuring.md).

## Custom population strategies

The [`PopulationGenerator`](../api/populations.md#hgp_lib.populations.generator.PopulationGenerator) creates the initial set of rules.
It uses a strategy pattern to allow different initialization methods.

When using [`BooleanGPConfig`](../api/configs.md#hgp_lib.configs.boolean_gp_config.BooleanGPConfig), pass a [`PopulationGeneratorFactory`](../api/populations.md#hgp_lib.populations.populations_factory.PopulationGeneratorFactory) rather than a [`PopulationGenerator`](../api/populations.md#hgp_lib.populations.generator.PopulationGenerator) directly.
Override `create_strategies` to use custom strategies.

```python
from hgp_lib.populations import (
    PopulationGeneratorFactory,
    RandomStrategy,
    BestLiteralStrategy,
)

class MyFactory(PopulationGeneratorFactory):
    def create_strategies(self, num_literals, evaluator):
        random = RandomStrategy(num_literals=num_literals)
        best = BestLiteralStrategy(
            num_literals=num_literals,
            evaluator=evaluator,
            sample_size=100,
            feature_size=None,
        )
        return [random, best]

factory = MyFactory(population_size=100)
```

The [`Evaluator`](../api/evaluation.md#hgp_lib.evaluation.backend.Evaluator) holds the population's training rows together with its scorer.
When the rows were merged into sample weights, both stay consistent:

- `evaluator.score(rules)` scores rules on all training rows, with the fast kernels bound to them.
- `evaluator.dataset` is the [`Dataset`](../api/evaluation.md#hgp_lib.evaluation.dataset.Dataset) of the training rows: `data`, `labels` and `sample_weight` (`None` when every row counts once).
- `evaluator.scorer(y_true, y_pred, sample_weight)` scores any subset of those rows.
  It passes `sample_weight` to the scoring function only when it is not `None`.

To score a random subset of the original rows, draw indices in `range(dataset.n_rows)` and select them with `Dataset.take`, which keeps the weights consistent:

```python
import numpy as np

dataset = evaluator.dataset
rows = np.random.choice(dataset.n_rows, 100, replace=False)
subset = dataset.take(rows)
column_score = evaluator.scorer(subset.labels, subset.data[:, 0], subset.sample_weight)
```

You can also create a [`PopulationGenerator`](../api/populations.md#hgp_lib.populations.generator.PopulationGenerator) directly for standalone use, outside of [`BooleanGPConfig`](../api/configs.md#hgp_lib.configs.boolean_gp_config.BooleanGPConfig).

```python
from hgp_lib.populations import PopulationGenerator, RandomStrategy

random_strategy = RandomStrategy(num_literals=10)
generator = PopulationGenerator(
    strategies=[random_strategy],
    population_size=100,
)
initial_population = generator.generate()
```

## Custom sampling strategies

A [`SamplingStrategy`](../api/populations.md#hgp_lib.populations.sampling.SamplingStrategy) splits the parent's training rows between child populations.
`sample` receives the parent's [`Dataset`](../api/evaluation.md#hgp_lib.evaluation.dataset.Dataset) and returns one [`SamplingResult`](../api/populations.md#hgp_lib.populations.sampling.SamplingResult) per child.
The parent's rows may be merged into sample weights, so count rows with `dataset.n_rows`, draw indices in `range(dataset.n_rows)`, and let `create_sampling_result` select them.

```python
import numpy as np
from hgp_lib.populations import SamplingStrategy

class FirstHalfSampling(SamplingStrategy):
    """Every child gets the first half of the original rows."""

    def sample(self, dataset, num_children):
        rows = np.arange(dataset.n_rows // 2)
        return [
            self.create_sampling_result(dataset, None, rows)
            for _ in range(num_children)
        ]
```

## Custom evaluation backends

An [`EvaluationBackend`](../api/evaluation.md#hgp_lib.evaluation.backend.EvaluationBackend) evaluates rules during training, validation and prediction.
A backend implements two methods:

- `predict(rule, data)` returns one boolean NumPy prediction per row of `data`.
- `_bind(dataset, scorer)` returns an [`Evaluator`](../api/evaluation.md#hgp_lib.evaluation.backend.Evaluator) that holds the backend's own representation of the data, with `score(rules)` and `confusion_matrix(rule)`.

The shared `bind` merges duplicate rows before calling `_bind` when the scorer allows it, so a backend only has to support `sample_weight` being `None` or integer row counts.
Backends hold options only, so they must be picklable to run in parallel benchmark workers.
`evaluator.dataset` must stay the NumPy [`Dataset`](../api/evaluation.md#hgp_lib.evaluation.dataset.Dataset), because population and sampling strategies read and sample it; keep the backend's own copy of the data in private attributes.
[`TorchBackend`](../api/evaluation.md#hgp_lib.evaluation.torch.backend.TorchBackend) is an example: it keeps the rows on its device and copies back only the counts.

## Custom mutations

A mutation subclasses [`Mutation`](../api/mutations.md#hgp_lib.mutations.base_mutation.Mutation) and edits a rule node in place inside `apply`.
The base class needs two flags that say whether the mutation can apply to literals, to operators, or to both.

The example below adds a `RandomNegate` mutation that flips a node's negation only some of the time, unlike the built-in [`NegateMutation`](../api/mutations.md#hgp_lib.mutations.literal_mutations.NegateMutation) that always flips it.
It works on both literals and operators, so both flags are `True`.

```python
import random
from hgp_lib.mutations import Mutation
from hgp_lib.rules import Rule

class RandomNegate(Mutation):
    def __init__(self, negate_p: float = 0.5):
        super().__init__(is_literal_mutation=True, is_operator_mutation=True)
        self.negate_p = negate_p

    def apply(self, rule: Rule):
        if random.random() < self.negate_p:
            rule.negated = not rule.negated
```

To use it, subclass [`MutationExecutorFactory`](../api/mutations.md#hgp_lib.mutations.mutation_factory.MutationExecutorFactory) and add the mutation in the relevant hook.
`create_literal_mutations` returns the mutations applied to literal nodes, and `create_operator_mutations` returns those applied to operator nodes.
Since `RandomNegate` handles both, add it to each.

```python
from hgp_lib.mutations import MutationExecutorFactory

class MyMutationFactory(MutationExecutorFactory):
    def create_literal_mutations(self, num_literals):
        return super().create_literal_mutations(num_literals) + (RandomNegate(),)

    def create_operator_mutations(self, num_literals):
        return super().create_operator_mutations(num_literals) + (RandomNegate(),)

mutation_factory = MyMutationFactory(mutation_p=0.1)
```

Pass `mutation_factory` to [`BooleanGPConfig`](../api/configs.md#hgp_lib.configs.boolean_gp_config.BooleanGPConfig) as shown in [Configuring HGP](configuring.md).
The factory builds the executor at runtime, once the number of features is known.

## Low-level use of BooleanGP

For full control over the training loop, use [`BooleanGP`](../api/algorithms.md#hgp_lib.algorithms.boolean_gp.BooleanGP) directly.
Training data is passed in the config, and `num_features` is derived from the data shape.
The number of features is then passed to the configured factories for runtime construction.

```python
from hgp_lib.configs import BooleanGPConfig
from hgp_lib.algorithms import BooleanGP
from hgp_lib.rules import ComplexityCheck

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
