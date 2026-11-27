# API Reference

Full reference for all public modules in the HGP library.

| Module | Description |
|--------|-------------|
| [Algorithms](algorithms.md) | [`BooleanGP`](algorithms.md#hgp_lib.algorithms.boolean_gp.BooleanGP), the core GP algorithm |
| [Configs](configs.md) | [`BooleanGPConfig`](configs.md#hgp_lib.configs.boolean_gp_config.BooleanGPConfig), [`TrainerConfig`](configs.md#hgp_lib.configs.trainer_config.TrainerConfig), [`BenchmarkerConfig`](configs.md#hgp_lib.configs.benchmarker_config.BenchmarkerConfig) |
| [Trainers](trainers.md) | [`GPTrainer`](trainers.md#hgp_lib.trainers.gp_trainer.GPTrainer), high-level training loop; [`BooleanRuleClassifier`](trainers.md#hgp_lib.trainers.boolean_rule_classifier.BooleanRuleClassifier), scikit-learn-style classifier on raw data |
| [Benchmarkers](benchmarkers.md) | [`GPBenchmarker`](benchmarkers.md#hgp_lib.benchmarkers.gp_benchmarker.GPBenchmarker), multi-run benchmarking |
| [Evaluation](evaluation.md) | [`predict`](evaluation.md#hgp_lib.evaluation.api.predict), [`score`](evaluation.md#hgp_lib.evaluation.api.score), scoring functions, [`NumpyBackend`](evaluation.md#hgp_lib.evaluation.numpy.backend.NumpyBackend), [`TorchBackend`](evaluation.md#hgp_lib.evaluation.torch.backend.TorchBackend) |
| [Rules](rules.md) | [`Rule`](rules.md#hgp_lib.rules.rules.Rule), [`Literal`](rules.md#hgp_lib.rules.literals.Literal), [`And`](rules.md#hgp_lib.rules.operators.And), [`Or`](rules.md#hgp_lib.rules.operators.Or) rule tree nodes, [`ComplexityCheck`](rules.md#hgp_lib.rules.constraints.ComplexityCheck) |
| [Postprocessing](postprocessing.md) | [`postprocess`](postprocessing.md#hgp_lib.postprocessing.api.postprocess) and [`postprocess_to_str`](postprocessing.md#hgp_lib.postprocessing.api.postprocess_to_str), simplify a rule and convert it to a canonical DNF or CNF |
| [Mutations](mutations.md) | Literal and operator mutations, [`MutationExecutor`](mutations.md#hgp_lib.mutations.mutation_executor.MutationExecutor) |
| [Crossover](crossover.md) | [`CrossoverExecutor`](crossover.md#hgp_lib.crossover.crossover_executor.CrossoverExecutor), subtree crossover |
| [Selections](selections.md) | Tournament, roulette selection strategies |
| [Populations](populations.md) | Population generation and sampling strategies |
| [Preprocessing](preprocessing.md) | [`StandardBinarizer`](preprocessing.md#hgp_lib.preprocessing.binarizer.StandardBinarizer), [`FeatureSpec`](preprocessing.md#hgp_lib.preprocessing.features.FeatureSpec), [`load_data`](preprocessing.md#hgp_lib.preprocessing.utils.load_data) |
| [Testing](testing.md) | [`check_backend`](testing.md#hgp_lib.testing.check_backend), checks a custom evaluation backend against the NumPy backend |
| [Results](results.md) | [`GenerationMetrics`](results.md#hgp_lib.results.generation.GenerationMetrics), [`PopulationHistory`](results.md#hgp_lib.results.history.PopulationHistory), [`RunResult`](results.md#hgp_lib.results.experiment.RunResult), [`ExperimentResult`](results.md#hgp_lib.results.experiment.ExperimentResult) |
