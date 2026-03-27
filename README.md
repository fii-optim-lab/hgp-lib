
# Hierarchical Genetic Programming Library

## Usage


### Data preparation

Data must be binarized before using a boolean GP. 
Binarization keeps boolean columns and transforms categorical and numeric features into bins given the configured setup.
For more information about binarization and custom configurations, see TODO (create a section about binarization and link it here).

Label aware binarization is usually employed to create class-aware bins for numerical columns.
The binarizer must be fit only on the training data to prevent data leakage!

Depending on usage, data preparation may vary (i.e. when doing k-fold cross-validation).

> **Note:** If you are using the **GPBenchmarker**, you do **not** need to binarize manually.
> The benchmarker handles per-fold binarization internally to prevent data leakage.
> See [Benchmarking Boolean GP](#benchmarking-boolean-gp) for details.

For manual training (GPTrainer / BooleanGP), binarize the data yourself:

```python
from hgp_lib.preprocessing import StandardBinarizer
from sklearn.model_selection import train_test_split


data, labels = ...  # Load data and labels

train_data, train_labels, test_data, test_labels = train_test_split(data, labels, random_state=42, stratify=labels)
train_data, train_labels, val_data, val_labels = train_test_split(train_data, train_labels, random_state=42, stratify=train_labels)

binarizer = StandardBinarizer(
    num_bins=5,  # Optional, num_bins is applied only to numerical features
)  
train_data = binarizer.fit_transform(train_data, train_labels)
val_data = binarizer.transform(val_data)
test_data = binarizer.transform(test_data)
```

**Config-based API**: The library uses dataclass configs (`BooleanGPConfig`, `TrainerConfig`, `BenchmarkerConfig`) for all main components. When you pass training data in a config, **the number of features is derived from the data** (`train_data.shape[1]`) and passed to the configured `PopulationGeneratorFactory` and `MutationExecutorFactory` at runtime, so you do not need to pass `num_literals` when using the default factories.

### Simple training

The snippet below will run a training with default hyperparameters. Use `BooleanGPConfig` and `TrainerConfig` to configure the run.

```python
from hgp_lib import BooleanGPConfig, TrainerConfig
from hgp_lib.trainers import GPTrainer

score_fn = ...  # My scoring function
num_epochs = 1000

gp_config = BooleanGPConfig(
    train_data=train_data,
    train_labels=train_labels,
    score_fn=score_fn,
)
config = TrainerConfig(
    gp_config=gp_config,
    num_epochs=num_epochs,
    val_data=val_data,
    val_labels=val_labels,
)
trainer = GPTrainer(config)
result = trainer.fit()
test_metrics = trainer.score(test_data, test_labels)
```

### Hyperparameter configuration

The `PopulationGeneratorFactory` and `MutationExecutorFactory` hold config-time parameters and build the actual `PopulationGenerator` / `MutationExecutor` at runtime (when `num_features` is known from the data). Subclass either factory to use custom strategies or mutations.

```python
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.mutations import MutationExecutorFactory
from hgp_lib.populations import PopulationGeneratorFactory, BestLiteralStrategy
from hgp_lib.selections import TournamentSelection, RouletteSelection, ParetoSelection
from hgp_lib.rules import Rule

max_rule_size = 100
population_size = 100
mutation_p = 0.1
crossover_p = 0.7
score_fn = ...  # My scoring function
regeneration = True
regeneration_patience = 100
num_epochs = 1000


def is_rule_valid(rule: Rule) -> bool:
    if len(rule) > max_rule_size:
        return False
    return True


# Population factory (default uses RandomStrategy)
population_factory = PopulationGeneratorFactory(population_size=population_size)

# Subclass to use custom strategies (e.g. BestLiteralStrategy)
class MyPopulationFactory(PopulationGeneratorFactory):
    def create_strategies(self, num_literals, score_fn, train_data, train_labels):
        return [BestLiteralStrategy(
            num_literals=num_literals,
            score_fn=score_fn,
            train_data=train_data,
            train_labels=train_labels,
            sample_size=0.1,  # Use 10% of data for evaluation
            feature_size=0.5, # Use 50% of features for evaluation
        )]

# Mutation factory (default uses standard literal and operator mutations)
mutation_factory = MutationExecutorFactory(mutation_p=mutation_p)

crossover_executor = CrossoverExecutor(
    crossover_p=crossover_p,  # Optional
    crossover_strategy="best",  # Optional. Default: `"random"`
    check_valid=is_rule_valid,  # Optional
    num_tries=2,  # Optional
)
selection = TournamentSelection(size=10, selection_probability=0.4)
```

* Population initialization is covered in [Population Generation](#population-generation).
* Mutation is covered in TODO.
* Crossover is covered in TODO.


### Population Generation

The `PopulationGenerator` creates the initial set of rules. It uses a strategy pattern to allow for different initialization methods.

When using `BooleanGPConfig`, pass a `PopulationGeneratorFactory` instead of a `PopulationGenerator` directly. The factory defers construction until the number of features is known from the data. Override `create_strategies` to use custom strategies:

```python
from hgp_lib.populations import (
    PopulationGeneratorFactory, PopulationGenerator,
    RandomStrategy, BestLiteralStrategy,
)

# Default factory: uses RandomStrategy, constructs the generator at runtime
factory = PopulationGeneratorFactory(population_size=100)

# Custom factory with BestLiteralStrategy
class MyFactory(PopulationGeneratorFactory):
    def create_strategies(self, num_literals, score_fn, train_data, train_labels):
        random = RandomStrategy(num_literals=num_literals)
        best = BestLiteralStrategy(
            num_literals=num_literals,
            score_fn=score_fn,
            train_data=train_data,
            train_labels=train_labels,
            sample_size=100,    # Evaluate on 100 random samples
            feature_size=None,  # Evaluate all features
        )
        return [random, best]

factory = MyFactory(population_size=100)
```

You can also create a `PopulationGenerator` directly for standalone use (outside of `BooleanGPConfig`):

```python
random_strategy = RandomStrategy(num_literals=10)
generator = PopulationGenerator(
    strategies=[random_strategy],
    population_size=100,
)
initial_population = generator.generate()
```

### Low level usage with fine control

Use `BooleanGPConfig` to configure the algorithm. Training data is passed in the config; **the number of features (`num_features`) is derived from the data shape** and passed to the configured `population_factory` and `mutation_factory` for runtime construction.

```python
from hgp_lib import BooleanGPConfig
from hgp_lib.algorithms import BooleanGP
from hgp_lib.mutations import MutationExecutorFactory
from hgp_lib.populations import PopulationGeneratorFactory

gp_config = BooleanGPConfig(
    train_data=train_data,
    train_labels=train_labels,
    score_fn=score_fn,
    population_factory=population_factory,  # Optional; default PopulationGeneratorFactory()
    mutation_factory=mutation_factory,  # Optional; default MutationExecutorFactory()
    crossover_factory=crossover_executor,
    selection=selection,
    check_valid=is_rule_valid,
    regeneration=regeneration,
    regeneration_patience=regeneration_patience,
)
gp_algo = BooleanGP(gp_config)

for i in range(num_epochs):
    train_metrics = gp_algo.step()  # No arguments; uses data from config
    if i % 100 == 0:
        val_metrics = gp_algo.validate_population(val_data, val_labels)
        print(f"Epoch {i} -> {val_metrics.best}, Population average: {val_metrics.population_scores.mean()}")

test_metrics = gp_algo.validate_population(test_data, test_labels, all_time_best=True)
print(f"Test result: Best: {test_metrics.best}, Population average: {test_metrics.population_scores.mean()}")
```

For lower level usage, consider inheriting from `BooleanGP` or `BooleanHGP` and overwriting `step`.

### Using a Boolean GP Trainer

Build a `BooleanGPConfig` and wrap it in a `TrainerConfig`. The trainer accepts only `TrainerConfig`.

```python
from hgp_lib import BooleanGPConfig, TrainerConfig
from hgp_lib.trainers import GPTrainer

gp_config = BooleanGPConfig(
    train_data=train_data,
    train_labels=train_labels,
    score_fn=score_fn,
    population_factory=population_factory,  # Optional; default PopulationGeneratorFactory()
    mutation_factory=mutation_factory,  # Optional; default MutationExecutorFactory()
    crossover_factory=crossover_executor,
    selection=selection,
    check_valid=is_rule_valid,
    regeneration=regeneration,
    regeneration_patience=regeneration_patience,
)
config = TrainerConfig(
    gp_config=gp_config,
    num_epochs=num_epochs,
    val_data=val_data,
    val_labels=val_labels,
    val_every=100,
)
trainer = GPTrainer(config)
result = trainer.fit()
test_metrics = trainer.score(test_data, test_labels)
# result.train_history.best_scores(), result.train_history.mean_scores(), etc.
```


### Benchmarking Boolean GP

The benchmarker runs multiple full runs (default 30), each with a stratified train/test split and k-fold CV on the training set. Results are aggregated across runs. Runs execute in parallel by default. The benchmarker accepts a `BenchmarkerConfig` containing a `TrainerConfig` template.

**Automatic binarization**: Pass raw (non-binarized) data as a `pandas.DataFrame` in `BenchmarkerConfig.data`. The benchmarker binarizes internally: for each fold, a fresh copy of the binarizer is fitted on the training fold (with labels for supervised binning) and used to transform the validation fold. The best fold's binarizer is then used to transform the held-out test set. This prevents data leakage across folds and between train/test splits.

By default a `StandardBinarizer()` is used. You can pass a custom binarizer (unfitted) via the `binarizer` parameter:

```python
from hgp_lib.preprocessing import StandardBinarizer

binarizer = StandardBinarizer(num_bins=10)  # must be unfitted
config = BenchmarkerConfig(
    data=data,
    labels=labels,
    trainer_config=trainer_config,
    binarizer=binarizer,  # None -> default StandardBinarizer(num_bins=5)
)
```

**Feature names**: The `BenchmarkResult` includes `feature_names_per_run`, a list of `Dict[int, str]` mappings (one per run) from literal indices to the binarized column names. Use this to display rules in human-readable form:

```python
best_idx = np.argmax(result.all_test_scores)
print(result.all_best_rules[best_idx].to_str(result.feature_names_per_run[best_idx]))
```

**Scorer Optimization**: The benchmarker can automatically optimize scorers per fold by deduplicating data and using sample weights. This significantly speeds up scoring for datasets with many duplicate rows. To use this feature, pass a base scorer (not pre-optimized) that accepts a `sample_weight` parameter, and set `optimize_scorer=True` in `BooleanGPConfig` (this is the default).

**Full example**:

```python
import numpy as np
import pandas as pd
from hgp_lib import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.benchmarkers import GPBenchmarker

def f1_score(predictions, labels, sample_weight=None):
    if sample_weight is None:
        tp = (predictions & labels).sum()
        pred_sum, label_sum = predictions.sum(), labels.sum()
    else:
        tp = np.dot(predictions & labels, sample_weight)
        pred_sum = np.dot(predictions, sample_weight)
        label_sum = np.dot(labels, sample_weight)
    if pred_sum == 0 or label_sum == 0:
        return 1.0 if pred_sum == label_sum == 0 else 0.0
    return 2 * tp / (pred_sum + label_sum)

data = pd.DataFrame(...)  # raw features as a DataFrame (bool / categorical / numeric)
labels = np.array(...)    # 1-D target array

# Create nested configs: BooleanGPConfig -> TrainerConfig -> BenchmarkerConfig
# Note: train_data/train_labels are not needed in gp_config here;
# the benchmarker will binarize and set them per fold.
gp_config = BooleanGPConfig(
    score_fn=f1_score,
    optimize_scorer=True,  # Default; enables scorer optimization per fold
)
trainer_config = TrainerConfig(
    gp_config=gp_config,
    num_epochs=num_epochs,
    val_every=100,
)
config = BenchmarkerConfig(
    data=data,
    labels=labels,
    trainer_config=trainer_config,
    num_runs=30,
    test_size=0.2,
    n_folds=5,
    n_jobs=-1,
)
benchmarker = GPBenchmarker(config)
result = benchmarker.fit()

# Aggregated metrics
print(f"Test score: {result.mean_test_score:.4f} ± {result.std_test_score:.4f}")

# Human-readable best rule
best_idx = np.argmax(result.all_test_scores)
print(result.all_best_rules[best_idx].to_str(result.feature_names_per_run[best_idx]))
```

**Important**: Do NOT pass pre-optimized scorers (e.g., from `optimize_scorer_for_data`) when using the benchmarker. Pre-optimized scorers have sample weights bound to the original data, which become invalid after train/test/fold splits. Either pass a base scorer with `optimize_scorer=True` (default), or use `optimize_scorer=False` for scorers without `sample_weight` support.


