# Performance benchmarks

Install the development dependencies before running the benchmarks:

```bash
python -m pip install -e '.[dev]'
```

## Creating artifacts

Create every currently supported artifact and skip files that already exist:

```bash
python benchmarks/create_artifacts.py --all
```

Create only rule artifacts, replacing existing files:

```bash
python benchmarks/create_artifacts.py --rule_artifacts --overwrite
```

Rule artifacts are committed under `benchmarks/artifacts/rule_evaluation` so benchmark runs do not evolve populations.
Each dataset and fold has one JSON file containing 100 rules.
The first serialized rule stores the binarized feature-name mapping shared by the population; later rules omit the duplicate mapping.
The rules are evolved for 500 generations with these fold policies:

| Fold | Complexity check | Complexity penalty |
| ---: | ---: | ---: |
| 0 | 50 | 0 |
| 1 | 100 | 0 |
| 2 | 100 | -0.001 (high reward) |
| 3 | 250 | -0.0005 (medium reward) |
| 4 | 500 | -0.001 (high reward) |

The complexity checks are passed to HGP's mutation and crossover logic; artifact generation does not filter or replace the final population.
Negative values reward larger rules through BooleanGP's existing regularized-selection formula and are assigned after configuration validation.

## Running benchmarks

Every saved run requires a machine and version identifier:

```bash
python benchmarks/benchmark.py \
  --all \
  --machine macbook-m2 \
  --version 2.1.0
```

An optional name distinguishes multiple results for the same machine and version:

```bash
python benchmarks/benchmark.py \
  --fast \
  --machine macbook-m2 \
  --version 2.1.0 \
  --name new-selection
```

Run one scenario by its identifier:

```bash
python benchmarks/benchmark.py \
  --scenario rule_evaluation.spambase.100_rules \
  --machine macbook-m2 \
  --version 2.1.0
```

Results are saved under `benchmarks/results` and are intended to be committed:

```text
<machine>-<version>.json
<machine>-<version>-<name>.json
```

Machine, version, and optional name are also stored inside the JSON, so reports do not depend on parsing the filename.

## Scenarios

Each dataset has a 500-epoch full-run scenario:

- `full_run.breast_cancer.500_epochs`
- `full_run.banknote_authentication.500_epochs`
- `full_run.diabetes.500_epochs`
- `full_run.spambase.500_epochs`
- `full_run.ionosphere.500_epochs`

Each dataset also has a fixed-population evaluation scenario:

- `rule_evaluation.breast_cancer.100_rules`
- `rule_evaluation.banknote_authentication.100_rules`
- `rule_evaluation.diabetes.100_rules`
- `rule_evaluation.spambase.100_rules`
- `rule_evaluation.ionosphere.100_rules`

A rule-evaluation scenario is one timed benchmark that evaluates all five folds sequentially.
It records one per-dataset runtime plus the mean and population standard deviation of the five fold scores.
Full-run scenarios still store one measurement per fold; reports aggregate their runtimes and scores by scenario.

Datasets are cached under `./data`; the four OpenML datasets are downloaded on first use.

## Comparing machines and versions

```bash
python benchmarks/compare_results.py
```

The report reads every JSON file under `benchmarks/results` and prints a separate table for each machine.
Rows are ordered by scenario and then version. Each row contains the scenario, version and optional result name, aggregate runtime, and `mean ± std` score when available.
Run performance comparisons under similar system load on each machine.
