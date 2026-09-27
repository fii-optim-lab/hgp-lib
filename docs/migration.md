# Migrating from 1.x to 2.0

In 2.0, rules only describe the logical structure. Evaluation backends evaluate them, during training and when you call `predict`.
Most code keeps working: deprecated names still work with a warning and will be removed in a future 2.x release.
The sections below list what to change, starting with what affects most users.

## Nothing to change for the default settings

With the default scorer and a single population, training gives the same rules and scores as 1.2 for the same seed, only faster.
Hierarchical runs can differ: child populations now keep their parent's sample weights, which fixes a bug where they were trained on a distorted copy of the data.
The configs now have defaults, so shorter code works too:

```python
from hgp_lib import BooleanRuleClassifier

clf = BooleanRuleClassifier()  # was: BooleanRuleClassifier(TrainerConfig(gp_config=BooleanGPConfig(), num_epochs=1000))
clf.fit(X_train, y_train)
```

`TrainerConfig()` uses `BooleanGPConfig()` and 1000 epochs, and `BenchmarkerConfig(data=X, labels=y)` uses `TrainerConfig()`.

## Evaluating a rule

`Rule.evaluate` is deprecated. Use `predict`, or `score` to also score the predictions.

```python
# 1.x
predictions = rule.evaluate(X_bin)
f1 = fast_f1_score(y, rule.evaluate(X_bin))

# 2.0
from hgp_lib.evaluation import predict, score

predictions = predict(rule, X_bin)
f1 = score(rule, X_bin, y)                    # fast_f1_score by default
accuracy = score(rule, X_bin, y, fast_accuracy_score)
```

[`predict`](api/evaluation.md#hgp_lib.evaluation.api.predict) and [`score`](api/evaluation.md#hgp_lib.evaluation.api.score) need the binarized boolean matrix, with the same columns the rule was trained on.
They raise a `TypeError` for a DataFrame or a non-boolean array.
In 1.x, `rule.evaluate` on an integer 0/1 array silently returned wrong results for negated literals.
For raw data, use `BooleanRuleClassifier.predict` or `GPBenchmarker.predict`, which binarize it first.

## Moved imports

| 1.x | 2.0 |
|---|---|
| `from hgp_lib.utils.metrics import fast_f1_score` | `from hgp_lib.evaluation import fast_f1_score` |
| `from hgp_lib.utils.metrics import confusion_matrix` | `from hgp_lib.evaluation import confusion_matrix` |
| `from hgp_lib.metrics import PopulationHistory` | `from hgp_lib.results import PopulationHistory` |
| `from hgp_lib.metrics.results import RunResult` | `from hgp_lib.results import RunResult` |
| `from hgp_lib.utils import ComplexityCheck` | `from hgp_lib.rules import ComplexityCheck` |

The old paths still work with a `DeprecationWarning`, except the submodules `hgp_lib.metrics.core`, `hgp_lib.metrics.history` and `hgp_lib.metrics.results`, which were renamed.
`hgp_lib.utils` is internal in 2.0.

## Custom scorers and `optimize_scorer`

`optimize_scorer` decides whether duplicate rows may be merged into integer sample weights, so rules are scored on fewer rows with the same results.
Its default changed from `True` to `None`:

| `optimize_scorer` | Built-in scorers | Custom `score_fn` |
|---|---|---|
| `None` (default) | merged | not merged |
| `True` | merged | merged if it accepts `sample_weight` |
| `False` | not merged | not merged |

A custom scorer that accepts `sample_weight` was merged automatically in 1.x. To keep that speed, opt in:

```python
from sklearn.metrics import balanced_accuracy_score

BooleanGPConfig(score_fn=balanced_accuracy_score, optimize_scorer=True)
```

With `optimize_scorer=True` and a scorer that does not accept `sample_weight`, 2.0 emits a `FutureWarning` and does not merge rows.
A future release will raise a `ValueError` instead, so pass `optimize_scorer=None` or `False` for such scorers.
See [Scorer optimization](guide/benchmarking.md#scorer-optimization).

## Low-memory evaluation and backends

`hgp_lib.rules.low_memory_operators` was removed, and the `HGP_LOW_MEMORY` environment variable is deprecated.
Choose the evaluation algorithm on the backend instead. It applies to training, validation and every `predict` method:

```python
# 1.x
# export HGP_LOW_MEMORY=1

# 2.0
from hgp_lib.evaluation import NumpyBackend

BooleanGPConfig(backend=NumpyBackend(low_memory=True))
```

`low_memory=True` is now the default, because it was also the faster algorithm in the benchmarks.
`HGP_LOW_MEMORY` still sets the default in 2.x (`"1"` for `True`, anything else for `False`), with a `FutureWarning`.
See [The NumPy backend](guide/rule-trees.md#the-numpy-backend) for all backend options.

## Custom population strategies

`PopulationGeneratorFactory.create_strategies` receives an evaluator instead of the scorer and the data.
The evaluator keeps the training rows and the scorer consistent when rows were merged into sample weights.

```python
# 1.x
class MyFactory(PopulationGeneratorFactory):
    def create_strategies(self, num_literals, score_fn, train_data, train_labels):
        return [BestLiteralStrategy(num_literals, score_fn, train_data, train_labels)]

# 2.0
class MyFactory(PopulationGeneratorFactory):
    def create_strategies(self, num_literals, evaluator):
        return [BestLiteralStrategy(num_literals, evaluator)]
```

In a custom strategy (see [Custom population strategies](guide/extending.md#custom-population-strategies)):

- `evaluator.score(rules)` scores rules on all training rows.
- `evaluator.dataset` has `data`, `labels` and `sample_weight` (`None` when every row counts once).
- `evaluator.scorer(labels, predictions, sample_weight)` scores a subset of the rows.
- To sample rows, draw indices in `range(evaluator.dataset.n_rows)` (original rows) and select them with `evaluator.dataset.take(rows)`.

```python
dataset = evaluator.dataset
rows = np.random.choice(dataset.n_rows, 100, replace=False)
subset = dataset.take(rows)
column_score = evaluator.scorer(subset.labels, subset.data[:, 0], subset.sample_weight)
```

## Custom sampling strategies

`SamplingStrategy.sample` receives a `Dataset` and returns `SamplingResult(dataset, feature_mapping)`.

```python
# 1.x
def sample(self, data, labels, num_children, sample_weight=None):
    ...
    return [self.create_sampling_result(data, labels, features, rows, sample_weight)]

# 2.0
def sample(self, dataset, num_children):
    ...
    return [self.create_sampling_result(dataset, features, rows)]
```

Count instances with `dataset.n_rows` and draw row indices in `range(dataset.n_rows)`: they refer to original rows, so sampling fractions keep their meaning when rows were merged.
Read the sampled rows from `result.dataset.data` and `result.dataset.labels`.
See [Custom sampling strategies](guide/extending.md#custom-sampling-strategies).

## Removed helpers and attributes

| Removed | Use instead |
|---|---|
| `SampleWeightScorer`, `optimize_scorers_for_data` | `EvaluationBackend.bind(dataset, resolve_scorer(score_fn, optimize))` |
| `transform_duplicates_to_sample_weight` | `Dataset(data, labels).deduplicate()` |
| `BooleanGP.score_fn`, `BooleanGP.original_score_fn` | `BooleanGP.scorer.fn` |
| `BooleanGP.train_cm`, `GPTrainer.val_cm` | `BooleanGP.evaluator.confusion_matrix(rule)`, `GPTrainer.val_evaluator.confusion_matrix(rule)` |
| `GPTrainer.score_fn`, `GPTrainer.val_score_fn` | `GPTrainer.gp_algo.scorer` (validation uses the same scorer) |
| `GPTrainer.val_data`, `GPTrainer.val_labels` | `GPTrainer.val_evaluator.dataset` |
| `BooleanGP(config, current_depth, sample_weight)` | `BooleanGP(config, current_depth, dataset)` |
| Custom `Rule` subclasses with their own `evaluate` | Not supported by backends; they support `Literal`, `And` and `Or` |

`BooleanGP.evaluate_population(data, labels, score_fn)` and `BooleanGP.evaluate_best(data, labels, score_fn=None)` are unchanged.

## Deprecation timeline

These work in 2.x with a warning and will be removed in a future 2.x release:

| Deprecated | Replacement | Warning |
|---|---|---|
| `Rule.evaluate(data)` | `hgp_lib.evaluation.predict(rule, data)` | `DeprecationWarning` |
| `hgp_lib.metrics` | `hgp_lib.results` | `DeprecationWarning` |
| `hgp_lib.utils.metrics` | `hgp_lib.evaluation` | `DeprecationWarning` |
| `hgp_lib.utils.ComplexityCheck` | `hgp_lib.rules.ComplexityCheck` | `DeprecationWarning` |
| `HGP_LOW_MEMORY` | `NumpyBackend(low_memory=...)` | `FutureWarning` |
| `optimize_scorer=True` with a scorer without `sample_weight` | `optimize_scorer=None` or `False` | `FutureWarning`, will raise |
