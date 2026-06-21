# API Reference

Full reference for all public modules in the HGP library.

| Module | Description |
|--------|-------------|
| [Algorithms](algorithms.md) | [`BooleanGP`](algorithms.md#hgp_lib.algorithms.boolean_gp.BooleanGP), the core GP algorithm |
| [Configs](configs.md) | [`BooleanGPConfig`](configs.md#hgp_lib.configs.boolean_gp_config.BooleanGPConfig), [`TrainerConfig`](configs.md#hgp_lib.configs.trainer_config.TrainerConfig), [`BenchmarkerConfig`](configs.md#hgp_lib.configs.benchmarker_config.BenchmarkerConfig) |
| [Trainers](trainers.md) | [`GPTrainer`](trainers.md#hgp_lib.trainers.gp_trainer.GPTrainer), high-level training loop |
| [Benchmarkers](benchmarkers.md) | [`GPBenchmarker`](benchmarkers.md#hgp_lib.benchmarkers.gp_benchmarker.GPBenchmarker), multi-run benchmarking |
| [Rules](rules.md) | [`Rule`](rules.md#hgp_lib.rules.rules.Rule), [`Literal`](rules.md#hgp_lib.rules.literals.Literal), [`And`](rules.md#hgp_lib.rules.operators.And), [`Or`](rules.md#hgp_lib.rules.operators.Or) rule tree nodes |
| [Mutations](mutations.md) | Literal and operator mutations, [`MutationExecutor`](mutations.md#hgp_lib.mutations.mutation_executor.MutationExecutor) |
| [Crossover](crossover.md) | [`CrossoverExecutor`](crossover.md#hgp_lib.crossover.crossover_executor.CrossoverExecutor), subtree crossover |
| [Selections](selections.md) | Tournament, roulette selection strategies |
| [Populations](populations.md) | Population generation and sampling strategies |
| [Preprocessing](preprocessing.md) | [`StandardBinarizer`](preprocessing.md#hgp_lib.preprocessing.binarizer.StandardBinarizer), [`load_data`](preprocessing.md#hgp_lib.preprocessing.utils.load_data) |
| [Metrics](metrics.md) | [`GenerationMetrics`](metrics.md#hgp_lib.metrics.core.GenerationMetrics), [`PopulationHistory`](metrics.md#hgp_lib.metrics.history.PopulationHistory), [`RunResult`](metrics.md#hgp_lib.metrics.results.RunResult), [`ExperimentResult`](metrics.md#hgp_lib.metrics.results.ExperimentResult) |
| [Utils](utils.md) | Validation helpers, scoring utilities |
