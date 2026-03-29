# HGP Library

A Python library for **Hierarchical Genetic Programming** (HGP) applied to
rule-based binary classification.

HGP evolves human-readable boolean rules (e.g. `And(age < 50, Or(income >= 30k, employed))`)
that classify data by combining genetic programming with hierarchical population
structures, crossover, mutation, and selection operators.

## Key features

- Evolve interpretable boolean rule trees from tabular data
- Hierarchical GP with configurable child populations and feature/instance sampling
- Built-in benchmarking with stratified k-fold CV and parallel execution
- Automatic binarization of numeric and categorical features
- Scorer optimization via data deduplication and sample weights
- Configurable mutations, crossover, and selection strategies
- Dataclass-based configuration for reproducibility

## Quick example

```python
import numpy as np
from hgp_lib.configs import BooleanGPConfig, TrainerConfig
from hgp_lib.trainers import GPTrainer

def accuracy(predictions, labels):
    return np.mean(predictions == labels)

train_data = ...   # 2D boolean numpy array
train_labels = ... # 1D integer numpy array

config = TrainerConfig(
    gp_config=BooleanGPConfig(
        score_fn=accuracy,
        train_data=train_data,
        train_labels=train_labels,
    ),
    num_epochs=500,
)
result = GPTrainer(config).fit()
```

## Navigation

- [Getting Started](getting-started.md) — installation, data preparation, training
- [API Reference](api/index.md) — full module documentation
- [Experimental Results](results.md) — benchmark results (placeholder)
