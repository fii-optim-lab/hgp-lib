# Getting Started

## Installation

```bash
pip install -e .
```

## A first run

`hgp_lib` evolves boolean rules that classify tabular data. The fastest way to
get a result is `GPBenchmarker`, which handles binarization, splitting, and
aggregation for you. Pass a raw `pandas.DataFrame` and a scoring function.

```python
import numpy as np
import pandas as pd
from hgp_lib.configs import BenchmarkerConfig, BooleanGPConfig, TrainerConfig
from hgp_lib.benchmarkers import GPBenchmarker

data = pd.DataFrame(...)  # raw features (bool / categorical / numeric)
labels = np.array(...)    # 1-D target array

gp_config = BooleanGPConfig(score_fn=my_score_fn)
trainer_config = TrainerConfig(gp_config=gp_config, num_epochs=1000, val_every=100)
config = BenchmarkerConfig(
    data=data,
    labels=labels,
    trainer_config=trainer_config,
    num_runs=30,
    n_folds=5,
    n_jobs=-1,
)
result = GPBenchmarker(config).fit()
print(result.best_rule.to_str(result.best_run.feature_names))
```

## Where to go next

- [Theory](theory.md): how the GP search works and why it beats greedy trees
- [Interpretability](interpretability.md): readable rules and explainable models
- [Data Preparation](guide/data-preparation.md): binarization and avoiding leakage
- [Training](guide/training.md): `GPTrainer` and run configuration
- [Benchmarking](guide/benchmarking.md): aggregated runs and scorer optimization
- [Configuring HGP](guide/configuring.md): factories and hierarchical GP settings
- [Extending HGP](guide/extending.md): custom strategies, mutations, and low-level use
- [Rule Trees](guide/rule-trees.md): the rule data structure and its speed optimizations
- [Experiments](experiments/index.md): reproducing dataset experiments
- [API Reference](api/index.md): full module documentation
