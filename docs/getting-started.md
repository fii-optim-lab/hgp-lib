# Getting Started

## Installation

```bash
pip install -e .
```

## Data preparation

Boolean GP operates on boolean data. The `StandardBinarizer` converts
numeric, categorical, and boolean columns into a purely boolean DataFrame.

```python
from hgp_lib.preprocessing import StandardBinarizer
from sklearn.model_selection import train_test_split

data, labels = ...  # pandas DataFrame + numpy array

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42,
)

binarizer = StandardBinarizer(num_bins=5)
train_bin = binarizer.fit_transform(train_data, train_labels)
test_bin = binarizer.transform(test_data)
```

When using `GPBenchmarker`, binarization is handled automatically per fold.

## Training with GPTrainer

```python
from hgp_lib.configs import BooleanGPConfig, TrainerConfig
from hgp_lib.trainers import GPTrainer

gp_config = BooleanGPConfig(
    score_fn=my_score_fn,
    train_data=train_bin.to_numpy(dtype=bool),
    train_labels=train_labels,
)
config = TrainerConfig(gp_config=gp_config, num_epochs=500)
result = GPTrainer(config).fit()
```

## Benchmarking with GPBenchmarker

```python
import pandas as pd
from hgp_lib.configs import BenchmarkerConfig
from hgp_lib.benchmarkers import GPBenchmarker

config = BenchmarkerConfig(
    data=data,           # raw DataFrame (not binarized)
    labels=labels,
    trainer_config=trainer_config,
    num_runs=30,
    n_folds=5,
    n_jobs=-1,
)
result = GPBenchmarker(config).fit()
print(result.test_scores)
```

## Hyperparameter tuning

Use the Optuna-based tuning script with a YAML search space config:

```bash
python scripts/optuna_hypertuning.py \
    --data-path data/PaySim.hdf \
    --study-name PaySim \
    --hp-config hyperparameter_configs/default.yaml \
    --n-trials 100
```
