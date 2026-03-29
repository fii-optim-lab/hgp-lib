# API Reference

Full reference for all public modules in the HGP library.

| Module | Description |
|--------|-------------|
| [Algorithms](algorithms.md) | `BooleanGP` — the core GP algorithm |
| [Configs](configs.md) | `BooleanGPConfig`, `TrainerConfig`, `BenchmarkerConfig` |
| [Trainers](trainers.md) | `GPTrainer` — high-level training loop |
| [Benchmarkers](benchmarkers.md) | `GPBenchmarker` — multi-run benchmarking |
| [Rules](rules.md) | `Rule`, `Literal`, `And`, `Or` — rule tree nodes |
| [Mutations](mutations.md) | Literal and operator mutations, `MutationExecutor` |
| [Crossover](crossover.md) | `CrossoverExecutor` — subtree crossover |
| [Selections](selections.md) | Tournament, roulette selection strategies |
| [Populations](populations.md) | Population generation and sampling strategies |
| [Preprocessing](preprocessing.md) | `StandardBinarizer`, `load_data` |
| [Metrics](metrics.md) | `GenerationMetrics`, `PopulationHistory`, `RunResult`, `ExperimentResult` |
| [Utils](utils.md) | Validation helpers, scoring utilities |
