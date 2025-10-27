


## Usage


### Data preparation

Data must be binarized before using a boolean GP. 
Binarization keeps boolean columns and transforms categorical and numeric features into bins given the configured setup.
For more information about binarization and custom configurations, see TODO (create a section about binarization and link it here).

Label aware binarization is usually employed to create class-aware bins for numerical columns.
The binarizer fit on the training data must be used on the validation and test data to prevent data leakage. 

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


### Simple training

The snippet below will run a training with default hyperparameters.

```python
from hgp_lib.populations import PopulationGenerator
from hgp_lib.trainers import GPTrainer


score_fn = ...  # My scoring function
num_epochs = 1000

trainer = GPTrainer(
    num_epochs=num_epochs,  # Optional
    score_fn=score_fn,  # Mandatory
    train_data=train_data,  # Mandatory
    train_labels=train_labels,  # Optional
    val_data=val_data,  # Optional
    val_labels=val_labels,  # Optional
)
trainer_metrics = trainer.fit()
test_metrics = trainer.evaluate(test_data, test_labels)
```

### Hyperparameter configuration


```python
from hgp_lib.mutations import MutationExecutor, standard_literal_mutations, standard_operator_mutations
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.selections import TournamentSelection, RoulleteSelection, ParetoSelection
from hgp_lib.populations import PopulationGenerator
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


population_generator = PopulationGenerator(
    score_fn=score_fn,  # Optional
    population_size=population_size,  # Optional
    strategy="random"  # Optional (will be extended)
)
mutation_executor = MutationExecutor(
    mutation_p=mutation_p,  # Optional
    literal_mutations=standard_literal_mutations,  # Optional
    operator_mutations=standard_operator_mutations,  # Optional
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

* Population initialization is covered in TODO.
* Mutation is covered in TODO.
* Crossover is covered in TODO.


### Low level usage with fine control

```python
from hgp_lib.algorithms import BooleanGP



gp_algo = BooleanGP(
    score_fn=score_fn,  # Mandatory
    
    population_generator=population_generator,  # Optional
    mutation_executor=mutation_executor,  # Optional
    crossover_executor=crossover_executor,  # Optional
    selection=selection,  # Optional
    
    regeneration=regeneration,  # Optional
    regeneration_patience=regeneration_patience,  # Optional
)

for i in range(num_epochs):
    train_metrics = gp_algo.step(train_data, train_labels)
    if i % 100 == 0:
        val_metrics = gp_algo.evaluate(val_data, val_labels, strategy="best")
        current_best = val_metrics["best"]
        print(f"Epoch {i} -> {current_best}")

test_metrics = gp_algo.evaluate(test_data, test_labels)
print(f"Test result: {test_metrics['best']}")
```

For lower level usage, consider inheriting from `BooleanGP` or `BooleanHGP` and overwriting `step`.

### Using a Boolean GP Trainer

```python
from hgp_lib.trainers import GPTrainer


trainer = GPTrainer(
    score_fn=score_fn,  # Mandatory
    num_epochs=num_epochs,  # Mandatory

    train_data=train_data,  # Mandatory
    train_labels=train_labels,  # Optional
    val_data=val_data,  # Optional
    val_labels=val_labels,  # Optional

    population_generator=population_generator,  # Optional
    mutation_executor=mutation_executor,  # Optional
    crossover_executor=crossover_executor,  # Optional
    selection=selection,  # Optional
    
    regeneration=regeneration,  # Optional
    regeneration_patience=regeneration_patience,  # Optional
    val_every=100,  # Optional
)
trainer_metrics = trainer.fit()
test_metrics = trainer.evaluate(test_data, test_labels)
```


### Benchmarking Boolean GP

```python
from hgp_lib.benchmarkers import GPBenchmarker


benchmarker = GPBenchmarker(
    score_fn=score_fn,  # Mandatory
    num_epochs=num_epochs,  # Mandatory

    train_data=train_data,  # Mandatory
    train_labels=train_labels,  # Optional
    val_data=val_data,  # Optional
    val_labels=val_labels,  # Optional

    population_generator=population_generator,  # Optional
    mutation_executor=mutation_executor,  # Optional
    crossover_executor=crossover_executor,  # Optional
    selection=selection,  # Optional
    
    regeneration=regeneration,  # Optional
    regeneration_patience=regeneration_patience,  # Optional
    val_every=100,  # Optional
    
    cv_splits=30,  # Optional
    val_size=0.3,  # Optional
)
benchmark_metrics = benchmarker.fit()
test_performance = benchmarker.evaluate(test_data, test_labels)
```


