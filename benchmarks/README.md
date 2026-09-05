# Performance benchmarks

Install the development dependencies before running the benchmarks:

```bash
python -m pip install -e '.[dev]'
```

## Running benchmarks

The fast suite runs the five breast-cancer folds:

```bash
python benchmarks/benchmark.py --fast
```

The complete suite runs five folds for each of the five datasets:

```bash
python benchmarks/benchmark.py --all --save 2.1.0
```

Run one dataset scenario by its identifier:

```bash
python benchmarks/benchmark.py \
  --scenario full_run.spambase.500_epochs \
  --save 2.1.0-spambase-change
```

`--save` defaults to `current`. Results are written to `benchmark-results/<name>.json`.
Names such as `2.1.0-new-selection` are supported.

## Scenarios

Each dataset has its own 500-epoch full-run scenario:

- `full_run.breast_cancer.500_epochs`
- `full_run.banknote_authentication.500_epochs`
- `full_run.diabetes.500_epochs`
- `full_run.spambase.500_epochs`
- `full_run.ionosphere.500_epochs`

Every scenario contains five deterministic stratified folds and therefore records five execution times and five test F1 scores.
Datasets are cached under `./data`; the four OpenML datasets are downloaded on first use.

## Comparing results

```bash
python benchmarks/compare.py
```

By default, the report shows the mean test score and population standard deviation across the five folds.
Use `--scores` to include every fold score:

```bash
python benchmarks/compare.py --scores
```

The comparison reads every JSON file under `benchmark-results`, orders names naturally, and places `current.json` last.
For every scenario, the first result containing that scenario is the comparison origin.
The report also shows total time and the timing and mean-score changes from that first result.
Run performance comparisons on the same machine and under similar system load.
