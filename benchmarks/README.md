# Performance benchmarks

Install the development dependencies before running the benchmarks:

```bash
python -m pip install -e '.[dev]'
```

## Creating artifacts

Create every missing artifact:

```bash
python benchmarks/create_artifacts.py --all
```

Create or replace only one artifact family:

```bash
python benchmarks/create_artifacts.py --rule_artifacts --overwrite
python benchmarks/create_artifacts.py --evaluation_artifacts --overwrite
```

### Evolved rule artifacts

Artifacts under `benchmarks/artifacts/rule_evaluation` contain the 100 evolved rules used by the dataset rule-evaluation scenarios.
The rules are evolved for 500 generations with these fold policies:

| Fold | Complexity check | Complexity penalty |
| ---: | ---: | ---: |
| 0 | 50 | 0 |
| 1 | 100 | 0 |
| 2 | 100 | -0.001 |
| 3 | 250 | -0.0005 |
| 4 | 500 | -0.001 |

### Default evaluation artifacts

Artifacts under `benchmarks/artifacts/evaluation_default` contain two deterministic sets of 100 rules:

- 100 rules containing exactly 100 literal nodes each.
- 100 rules containing exactly 1,000 literal nodes each.

Every root is a random `And` or `Or`. Operators have between two and six children, and rule depth does not exceed 20.

## Running benchmarks

Every saved run requires a machine and version identifier:

```bash
python benchmarks/benchmark.py \
  --all \
  --machine macbook-m2 \
  --version 2.1.0
```

An optional name distinguishes comparable variants for the same machine and version:

```bash
python benchmarks/benchmark.py \
  --fast \
  --machine macbook-m2 \
  --version 2.1.0 \
  --name new-selection
```

Run one scenario family by its identifier:

```bash
python benchmarks/benchmark.py \
  --scenario scoring_default \
  --machine macbook-m2 \
  --version 2.1.0
```

Results are committed under `benchmarks/results` as:

```text
<machine>-<version>.json
<machine>-<version>-<name>.json
```

Machine, version, and optional name are also stored inside the JSON.

## Scenarios

### Full runs

Each dataset has a `full_run.<dataset>.500_epochs` scenario. These retain one measurement per fold; reports aggregate the five runtimes and scores.

### Dataset rule evaluation

Each dataset has a `rule_evaluation.<dataset>.100_rules` scenario. One timed call evaluates all five fold populations sequentially and stores the five fold scores as mean ± population standard deviation.

### Default scoring

The `scoring_default` family scores 100 deterministic predictions with 100 independent sample-weight arrays using `fast_f1_score`:

- `scoring_default_1_000`
- `scoring_default_10_000`
- `scoring_default_100_000`
- `scoring_default_1_000_000`

### Default rule evaluation

The `evaluation_default` family evaluates 100 fixed rules against deterministic boolean matrices:

- `evaluation_default_100_literals_1_000_samples`
- `evaluation_default_100_literals_10_000_samples`
- `evaluation_default_1_000_literals_1_000_samples`
- `evaluation_default_1_000_literals_10_000_samples`

## Comparing machines and versions

Print a Markdown report:

```bash
python benchmarks/compare_results.py
```

Write it directly to a publishable file:

```bash
python benchmarks/compare_results.py --output benchmark-report.md
```

The report has separate machine sections and scenario tables. Versions are compared only when machine, scenario, and optional result name match. Each version shows runtime, relative speed versus the first available version, and score mean ± standard deviation when available.
