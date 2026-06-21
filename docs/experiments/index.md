# Experiments

This section documents how to reproduce the experiments run with the HGP library on the datasets used in our evaluation.

Each page covers, where applicable:

- how to obtain the raw data
- how to preprocess it into the HDF format consumed by the library
- how to run a benchmark with `scripts/run_benchmark.py`

## Datasets

- [PMLB](pmlb.md): a suite of binary classification benchmarks, used to evaluate HGP across many small to medium datasets.
- [PaySim](paysim.md): synthetic mobile-money transactions for fraud detection, large and strongly imbalanced.
- [AEAC](aeac.md): Amazon employee access requests, where readable rules double as access-control policies.

## Conventions

All scripts use a consistent `--data_path` argument to point at the dataset.
Preprocessing scripts accept `--data_path` as the folder that holds the raw input and receives the generated `.hdf` files.
`scripts/run_benchmark.py` and `scripts/optuna_hypertuning.py` take `--data_path` as the path to a single preprocessed `.hdf` file.
