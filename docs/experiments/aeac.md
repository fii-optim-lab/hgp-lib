# AEAC (Amazon Employee Access Challenge)

The [Amazon Employee Access Challenge](https://www.kaggle.com/c/amazon-employee-access-challenge) (AEAC) is a binary classification dataset.
The task is to predict whether an employee should be granted access to a given resource based on their role attributes.

TODO: Describe policy generation. Unlike standard classifiers, HGP produces
readable boolean rules that can be turned into access-control policies for a
resource. Explain this use case and reference the paper once it is available.

## Data preparation

1. Download the Amazon Employee Access Challenge dataset from [Kaggle](https://www.kaggle.com/c/amazon-employee-access-challenge).
2. Place `train.csv` in the `./data` folder.
3. Run the preprocessing script:

   ```bash
   python scripts/preprocess/aeac_preprocess.py --data_path data
   ```

The preprocessing script is located at [`scripts/preprocess/aeac_preprocess.py`](https://github.com/fii-optim-lab/hgp-lib/blob/master/scripts/preprocess/aeac_preprocess.py).

### What the script does

- Reads `train.csv` from the `--data_path` folder.
- Renames the `ACTION` column to `target`, which is the label expected by the library.
- Casts every column to the pandas `category` dtype.
- Writes the full dataset to `data/AEAC.hdf`.
- Builds a per-resource "unseen class value" (UCV) dataset for a fixed list of resources (`75078`, `25993`, `79092`, `4675`).

Each UCV dataset contains the rows for that resource plus the users that never appear for it, added as negative (`target == 0`) examples.
These are written to `data/AEAC_<resource>_ucv.hdf`.

The `--data_path` argument defaults to `data`.
It is used both as the folder that holds `train.csv` and as the destination for the generated `.hdf` files.

## Running a benchmark

Once preprocessing is done, benchmark the Boolean GP on one of the generated HDF files with `scripts/run_benchmark.py`.

```bash
python scripts/run_benchmark.py --data_path data/AEAC_79092_ucv.hdf
```

Quick run with fewer runs and folds for a smoke test:

```bash
python scripts/run_benchmark.py \
    --data_path data/AEAC_79092_ucv.hdf \
    --num_runs 5 \
    --n_folds 3 \
    --num_epochs 500
```

Hierarchical GP variant:

```bash
python scripts/run_benchmark.py \
    --data_path data/AEAC_79092_ucv.hdf \
    --max_depth 1 \
    --num_child_populations 3
```

See `python scripts/run_benchmark.py --help` for the full list of options.
