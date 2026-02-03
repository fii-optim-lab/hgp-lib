
# Hierarchical Genetic Programming Library

## Usage


### Data preparation

Data must be binarized before using a boolean GP. 
Binarization keeps boolean columns and transforms categorical and numeric features into bins given the configured setup.
For more information about binarization and custom configurations, see TODO (create a section about binarization and link it here).

Label aware binarization is usually employed to create class-aware bins for numerical columns.
The binarizer must be fit only on the training data to prevent data leakage!

Depending on usage, data preparation may very (i.e. when doing k-fold cross-validation).

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

**Config-based API**: The library uses dataclass configs (`BooleanGPConfig`, `TrainerConfig`, `BenchmarkerConfig`) for all main components. When you pass training data in a config, **the number of features is derived from the data** (`train_data.shape[1]`), so you do not need to pass `num_literals` when using default population generator and mutation executor.

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


```python
from hgp_lib.mutations import MutationExecutor, create_standard_literal_mutations, create_standard_operator_mutations
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.selections import TournamentSelection, RouletteSelection, ParetoSelection
from hgp_lib.populations import PopulationGenerator, RandomStrategy, BestLiteralStrategy
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


num_features = train_data.shape[1]  # Derive from data; no need to pass num_literals into config when using defaults
literal_mutations = create_standard_literal_mutations(num_features)
operator_mutations = create_standard_operator_mutations(num_features)

random_strategy = RandomStrategy(num_literals=num_features)
best_literal_strategy = BestLiteralStrategy(
    num_literals=num_features,
    score_fn=score_fn,
    train_data=train_data,
    train_labels=train_labels,
    sample_size=0.1,  # Use 10% of data for evaluation
    feature_size=0.5  # Use 50% of features for evaluation
)

population_generator = PopulationGenerator(
    strategies=[random_strategy, best_literal_strategy],
    population_size=population_size,
    weights=[0.8, 0.2]  # 80% Random, 20% Best Literal
)

mutation_executor = MutationExecutor(
    literal_mutations=literal_mutations,  # Mandatory
    operator_mutations=operator_mutations,  # Mandatory
    mutation_p=mutation_p,  # Optional
    check_valid=is_rule_valid,  # Optional
    num_tries=10,  # Optional
)
crossover_executor = CrossoverExecutor(
    crossover_p=crossover_p,  # Optional
    crossover_strategy="best",  # Optional. Default: "random"
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

```python
from hgp_lib.populations import PopulationGenerator, RandomStrategy, BestLiteralStrategy

# Simple Random Strategy
# Generates rules with one operator (And/Or) and two random literals
random_strategy = RandomStrategy(num_literals=10)

# Best Literal Strategy
# Evaluates all single literals on a subset of data and picks the best one
best_literal_strategy = BestLiteralStrategy(
    num_literals=10,
    score_fn=my_score_fn,
    train_data=X_train,
    train_labels=y_train,
    sample_size=100,  # Evaluate on 100 random samples
    feature_size=None # Evaluate all features
)

# Combine strategies
generator = PopulationGenerator(
    strategies=[random_strategy, best_literal_strategy],
    population_size=100,
    weights=[0.7, 0.3] # 70% of rules from Random, 30% from Best Literal
)

initial_population = generator.generate()
```

### Low level usage with fine control

Use `BooleanGPConfig` to configure the algorithm. Training data is passed in the config; **the number of features (num_features) is derived from the data shape**, so you do not need to pass `num_literals` separately when using default population generator and mutation executor.

```python
from hgp_lib import BooleanGPConfig
from hgp_lib.algorithms import BooleanGP

gp_config = BooleanGPConfig(
    train_data=train_data,
    train_labels=train_labels,
    score_fn=score_fn,
    population_generator=population_generator,  # Optional; default uses num_features from data
    mutation_executor=mutation_executor,  # Optional; default uses num_features from data
    crossover_executor=crossover_executor,
    selection=selection,
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
    population_generator=population_generator,
    mutation_executor=mutation_executor,
    crossover_executor=crossover_executor,
    selection=selection,
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

**Scorer Optimization**: The benchmarker can automatically optimize scorers per fold by deduplicating data and using sample weights. This significantly speeds up scoring for datasets with many duplicate rows. To use this feature, pass a base scorer (not pre-optimized) that accepts a `sample_weight` parameter, and set `optimize_scorer=True` in `BooleanGPConfig` (this is the default).

```python
import numpy as np
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

# Create nested configs: BooleanGPConfig -> TrainerConfig -> BenchmarkerConfig
# Note: train_data/train_labels are not needed in gp_config here;
# the benchmarker will set them per fold from the full dataset.
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
# result.run_metrics, result.mean_test_score, result.std_test_score,
# result.mean_best_val_score, result.std_best_val_score, result.all_test_scores, result.all_best_rules
```

**Important**: Do NOT pass pre-optimized scorers (e.g., from `optimize_scorer_for_data`) when using the benchmarker. Pre-optimized scorers have sample weights bound to the original data, which become invalid after train/test/fold splits. Either pass a base scorer with `optimize_scorer=True` (default), or use `optimize_scorer=False` for scorers without `sample_weight` support.


