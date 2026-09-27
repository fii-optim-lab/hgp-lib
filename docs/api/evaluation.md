# Evaluation

Turning rules into predictions, scores and confusion matrices.

`predict` and `score` evaluate one rule on binarized data, as given.
For raw (non-binarized) data, use
[`BooleanRuleClassifier.predict`](trainers.md#hgp_lib.trainers.boolean_rule_classifier.BooleanRuleClassifier)
or [`GPBenchmarker.predict`](benchmarkers.md#hgp_lib.benchmarkers.gp_benchmarker.GPBenchmarker).

## Rules on data

::: hgp_lib.evaluation.api.predict

::: hgp_lib.evaluation.api.score

## Scoring functions

::: hgp_lib.evaluation.scorers.fast_f1_score

::: hgp_lib.evaluation.scorers.fast_accuracy_score

::: hgp_lib.evaluation.scorers.confusion_matrix

::: hgp_lib.evaluation.scorers.accepts_sample_weight

## Backends

::: hgp_lib.evaluation.numpy.backend.NumpyBackend

::: hgp_lib.evaluation.torch.backend.TorchBackend

::: hgp_lib.evaluation.backend.EvaluationBackend

::: hgp_lib.evaluation.backend.Evaluator

## Data and scorers bound during training

::: hgp_lib.evaluation.dataset.Dataset

::: hgp_lib.evaluation.scorers.Scorer

::: hgp_lib.evaluation.scorers.resolve_scorer
