# PaySim

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) is a synthetic dataset that simulates mobile-money transactions for fraud detection.
The task is to predict whether a transaction is fraudulent.

This dataset is large.
It has 6,362,620 transactions, of which only 8,213 are fraudulent and 6,354,407 are not.
The strong class imbalance makes it a good test for fraud detection, and the size makes runs slower than on the smaller benchmark datasets.

## Data preparation

Download `PaySim.csv` from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it in the `./data` folder.
Then run the preprocessing script at [`scripts/preprocess/paysim_preprocess.py`](https://github.com/fii-optim-lab/hgp-lib/blob/master/scripts/preprocess/paysim_preprocess.py).

```bash
python scripts/preprocess/paysim_preprocess.py --data_path data
```

The script renames the `isFraud` column to `target`, adds two boolean features for external origin and destination accounts, and writes `data/PaySim.hdf`.

## Running a benchmark

Benchmark the Boolean GP on the preprocessed dataset with `scripts/run_benchmark.py`.

```bash
python scripts/run_benchmark.py --data_path data/PaySim.hdf
```

Because the dataset is large, a quick run with fewer runs, folds, and epochs is useful for a first check.

```bash
python scripts/run_benchmark.py \
    --data_path data/PaySim.hdf \
    --num_runs 5 \
    --n_folds 3 \
    --num_epochs 500
```

See `python scripts/run_benchmark.py --help` for the full list of options.

## Hyperparameter tuning

Tune the dataset with the Optuna script.

```bash
python scripts/optuna_hypertuning.py \
    --data_path data/PaySim.hdf \
    --study_name PaySim \
    --hp_config hyperparameter_configs/default.yaml \
    --n_trials 100 \
    --artifact_dir ./artifacts
```
