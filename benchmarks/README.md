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

Each dataset has a `rule_evaluation.<dataset>.100_rules` scenario. One timed call evaluates all five fold populations sequentially and stores the five fold scores as mean +/- population standard deviation.

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

Print a report for every available machine:

```bash
python benchmarks/compare_results.py
```

Select machines by repeating `--machine`:

```bash
python benchmarks/compare_results.py \
  --machine macbook-m2 \
  --machine workstation-linux
```

Write the report directly to a publishable Markdown file:

```bash
python benchmarks/compare_results.py --output benchmark-report.md
```

The report is titled `hgp-lib performance report` and contains one table per scenario and optional result name.
All selected machines share the same table, with Time and vs previous subcolumns.
A Result subcolumn is shown for the first machine when scores are available; another machine receives one only when its result differs or has no matching first-machine result.
Versions are compared only within the same machine, scenario, and result name, and each timing change is relative to that machine's previous available version.
Generated labels and placeholders use ASCII characters for reliable display across platforms.

## Backend benchmarks

`benchmark_backends.py` compares the option combinations of the evaluation backends on one version, to choose the most suitable one.
It is separate from the default benchmark: `benchmark.py` ignores `benchmarks/backends`, and the backend scenarios are not part of the version comparison.

```bash
python benchmarks/benchmark_backends.py \
  --machine macbook-m2 \
  --version 2.1.0
```

`--machine` and `--version` are required; `--name` is optional, as for `benchmark.py`.
Every scenario runs for every combination, and the results are saved under `benchmarks/results/backends` as:

```text
<machine>-<version>.json
<machine>-<version>.md
```

The Markdown report ranks the combinations by the geometric mean of their time relative to the fastest combination in each scenario, then lists every scenario from fastest to slowest.
Backend options only change speed and memory use, so each scenario's result must be the same for every combination; the report lists the scenarios where it is not.

### Backend options

The backends and the values of their options are hardcoded in `BACKEND_OPTIONS` in `benchmarks/backends/test_backends.py`.
Every combination of the values is benchmarked:

| Backend | Option | Values |
| --- | --- | --- |
| `NumpyBackend` | `order` | `"F"`, `"C"` |
| `NumpyBackend` | `low_memory` | `True`, `False` |
| `NumpyBackend` | `batched` | `False`, `True` |
| `TorchBackend` | `device` | `"cpu"`, plus `"cuda"` and `"mps"` when available |
| `TorchBackend` | `batched` | `False`, `True` |

`TorchBackend` is only benchmarked when PyTorch is installed (`pip install "hgp-lib[torch]"`).
`batch_size` keeps its default (`None`, one batch per population) for both backends.
To benchmark a new backend or option value, add it to `BACKEND_OPTIONS`.

### Backend scenarios

- `population_scoring.<dataset>` scores the 100 evolved rules of each of the five folds on their training rows, merged into sample weights as during training.
- `population_scoring.<dataset>.x10_rows` scores the same rules on the training rows repeated 10 times, without merging, as for a larger dataset without duplicate rows.
- `random_rule_scoring.<literals>_literals.<rows>_rows` scores the 100 random rules of the default evaluation artifacts on random data.
- `predict.100_literals.10_000_rows` evaluates the 100-literal random rules once on row-major data, as `hgp_lib.evaluation.predict` does. `order` and `batched` do not apply to it.
- `training.<dataset>.100_generations` trains a flat population of 100 rules for 100 generations on the first fold of `breast_cancer` and `spambase`.
