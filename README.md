


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

binarizer = StandardBinarizer(num_bins=5)
train_data = binarizer.fit(train_data, train_labels)
val_data = binarizer.transform(val_data)
test_data = binarizer.transform(test_data)
```


### Hyperparameter configuration



* Configuring population initialization is covered in TODO.
* Configuring mutations is covered in TODO.
* Configuring crossover is covered in TODO.
```python
from hgp_lib.mutations import MutationExecutor, default_mutations
from hgp_lib.crossover import CrossoverExecutor
from hgp_lib.populations import PopulationGenerator
from hgp_lib.rules import Rule, tautology_or_contradiction


max_rule_size = 100
population_size = 100
mutation_p = 0.1
crossover_p = 0.7
score_fn = ...  # My scorig function
regeneration = True
regeneration_patience = 100
num_epochs = 1000


def is_rule_valid(rule: Rule) -> bool:
    if len(rule) > max_rule_size:
        return False
    if tautology_or_contradiction(rule):
        return False
    return True
 

population_generator = PopulationGenerator(
    data=train_data, 
    labels=train_labels, 
    population_size=population_size, 
    strategy="random"
)
mutation_executor = MutationExecutor(
    mutations=default_mutations,
    mutation_p=mutation_p,
    check_valid=is_rule_valid,
    num_tries=10,
)
crossover_executor = CrossoverExecutor(
    crossover_p=crossover_p,
    crossover_strategy="best",  # Can be "best" or "random"
    check_valid=is_rule_valid,
    num_tries=2,
)
```

### Low level usage with fine control

```python
from hgp_lib.algorithms import BooleanGP


gp_algo = BooleanGP(
    population_generator=population_generator,
    mutation_executor=mutation_executor,
    crossover_executor=crossover_executor,
    score_fn=score_fn,
    regeneration=regeneration,
    regeneration_patience=regeneration_patience,
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

gp_algo = BooleanGP(
    population_generator=population_generator,
    mutation_executor=mutation_executor,
    crossover_executor=crossover_executor,
    score_fn=score_fn,
    regeneration=regeneration,
    regeneration_patience=regeneration_patience,
)

trainer = GPTrainer(
    gp_algo=gp_algo,
    train_data=train_data, 
    val_data=val_data,
    val_labels=val_labels,
    train_labels=train_labels,
    num_iterations=num_iterations,
    val_every=100,
)
trainer_metrics = trainer.train()
test_metrics = trainer.evaluate(test_data, test_labels)
```


