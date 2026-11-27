# Migrating from 1.x to 2.0

In 2.0, rules only describe the logical structure. Evaluation backends evaluate them, during training and when you call `predict`.
Most code keeps working: deprecated names still work with a warning and will be removed in a future 2.x release.
The sections below list what to change, starting with what affects most users.

## Results with the default settings

Scores are `float32` by default in 2.0 (`float64` in 1.x).
They are the 1.x scores rounded to `float32`, so runs can diverge from 1.x where two rules only differ beyond `float32` precision.
With `NumpyBackend(dtype="float64")`, a single population gives the same rules and scores as 1.2 for the same seed.
Hierarchical runs can differ: child populations now keep their parent's sample weights, which fixes a bug where they were trained on a distorted copy of the data.

The configs now have defaults, so shorter code works too:

```python
from hgp_lib import BooleanRuleClassifier

clf = BooleanRuleClassifier()  # was: BooleanRuleClassifier(TrainerConfig(gp_config=BooleanGPConfig(), num_epochs=1000))
clf.fit(X_train, y_train)
```

`TrainerConfig()` uses `BooleanGPConfig()` and 1000 epochs, and `BenchmarkerConfig(data=X, labels=y)` uses `TrainerConfig()`.

## Data and labels

Binarized data must be a boolean NumPy array everywhere: `BooleanGPConfig.train_data`, `TrainerConfig.val_data`, `GPTrainer.predict`, `BooleanGP.evaluate_population` and `evaluate_best`, and `hgp_lib.evaluation.predict` and `score`.
In 1.x, integer 0/1 data was accepted; 2.0 raises a `TypeError` for it.
Binarizer output is boolean, so `binarizer.transform(X).to_numpy()` needs no `dtype=bool`.

Labels can be booleans, or integers 0 and 1, which are converted once.
Float labels raise a `TypeError`, and integer labels other than 0 and 1 raise a `ValueError`.
In 1.x, float labels failed on the first scoring call.

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
`score` scores through the backend, so its result has the backend's `dtype` precision.
In 1.x, `rule.evaluate` on an integer 0/1 array returned wrong results for a negated literal on its own.
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

### Pickled 1.x results

`RunResult`, `ExperimentResult` and `PopulationHistory` objects pickled with 1.x do not load in 2.0: the pickles reference the modules `hgp_lib.metrics.core`, `hgp_lib.metrics.history` and `hgp_lib.metrics.results`, which no longer exist.
The same holds for rules pickled with `HGP_LOW_MEMORY=1`, which reference the removed `hgp_lib.rules.low_memory_operators`.
The classes have the same fields in `hgp_lib.results`, so the data can be recreated from the old objects in a 1.x environment, for example as JSON.

## Custom scorers and `optimize_scorer`

`BooleanGPConfig.score_fn` defaults to `fast_f1_score` instead of `None`.
Passing `score_fn=None` explicitly raises a `TypeError`; leave it unset for the default.

`optimize_scorer` decides whether duplicate rows may be merged into integer sample weights, so rules are scored on fewer rows with the same results.
Its default changed from `True` to `None`:

| `optimize_scorer` | Built-in scorers | Custom `score_fn` |
|---|---|---|
| `None` (default) | merged | not merged |
| `True` | merged | merged; the scorer must accept `sample_weight` |
| `False` | not merged | not merged |

A custom scorer that accepts `sample_weight` was merged automatically in 1.x. To keep merging its rows, opt in:

```python
from sklearn.metrics import balanced_accuracy_score

BooleanGPConfig(score_fn=balanced_accuracy_score, optimize_scorer=True)
```

With `optimize_scorer=True` and a scorer that does not accept `sample_weight`, 2.0 raises a `ValueError` that explains the fix; 1.x silently did not merge.
A custom scorer must return a finite float, never NaN.
With `TorchBackend`, a custom scorer receives tensors.
See [Scorer optimization](guide/benchmarking.md#scorer-optimization) and [Custom scorers](guide/rule-trees.md#custom-scorers).

## Low-memory evaluation and backends

`hgp_lib.rules.low_memory_operators` was removed, and the `HGP_LOW_MEMORY` environment variable is deprecated and ignored: it only emits a `FutureWarning`.
Choose the evaluation implementation on the backend instead. It applies to training, validation and every `predict` method:

```python
# 1.x
# export HGP_LOW_MEMORY=1

# 2.0
from hgp_lib.evaluation import NumpyBackend

BooleanGPConfig(backend=NumpyBackend(evaluation="inplace"))
```

`evaluation="inplace"` is the algorithm of 1.x `low_memory_operators`, and `evaluation="block"` is the algorithm of the 1.x default operators.
See [The NumPy backend](guide/rule-trees.md#the-numpy-backend) for all backend options.

## Custom population strategies

`PopulationGeneratorFactory.create_strategies` receives an evaluator instead of the scorer and the data.
The evaluator keeps the training rows and the scoring function consistent when rows were merged into sample weights.

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

- `evaluator.score(rules)` scores rules on all training rows; `evaluator.backend.to_numpy` converts the scores to NumPy.
- `evaluator.subset(rows).score(rules)` scores rules on a subset of the original rows, drawn from `range(evaluator.dataset.n_rows)`.
- `evaluator.dataset` has `data`, `labels` and `sample_weight` (`None` when every row counts once), in the backend's array type.

```python
rows = np.random.choice(evaluator.dataset.n_rows, 100, replace=False)
scores = evaluator.backend.to_numpy(evaluator.subset(rows).score([Literal(value=0)]))
```

## Custom sampling strategies

`SamplingStrategy.sample` receives the parent's dataset and returns `SamplingResult(dataset, feature_mapping)`.

```python
# 1.x
def sample(self, data, labels, num_children):
    ...
    return [self.create_sampling_result(data, labels, features, rows)]

# 2.0
def sample(self, dataset, num_children):
    ...
    return [self.create_sampling_result(dataset, features, rows)]
```

Count instances with `dataset.n_rows` and draw row indices in `range(dataset.n_rows)`: they refer to original rows, so sampling fractions keep their meaning when rows were merged.
Count features with `dataset.n_features`.
`SamplingResult` has a `dataset` instead of `data` and `labels`.
See [Custom sampling strategies](guide/extending.md#custom-sampling-strategies).

## Custom binarizer hooks

The `_fit_boolean`, `_fit_categorical` and `_fit_numeric` hooks of a `StandardBinarizer` subclass return `(FeatureSpec, values)` pairs instead of `(name, values)` pairs.
The column name is `str(spec)`.

```python
# 1.x
return [(f"{column} is even", values % 2 == 0)]

# 2.0
from hgp_lib.preprocessing import BooleanFeature

return [(BooleanFeature(f"{column} is even"), values % 2 == 0)]
```

Use the spec that says what the column tests (`BooleanFeature`, `CategoryFeature`, `BinFeature` or `MissingFeature`), so that [postprocessing](guide/postprocessing.md#features) can merge the columns of one original column.
A `BooleanFeature` named after the output column is always correct; its columns are then treated as independent.
The private `_format_numeric_bin_name` is removed: `str(BinFeature(column, left, right))` gives the same name.

## Removed helpers and attributes

| Removed | Use instead |
|---|---|
| `SampleWeightScorer`, `optimize_scorers_for_data` | `backend.bind(dataset, score_fn, merge_rows=True)` |
| `transform_duplicates_to_sample_weight` | `Dataset(data, labels).deduplicate()` |
| `BooleanGP.score_fn`, `BooleanGP.original_score_fn` | `BooleanGP.evaluator.score_fn` |
| `BooleanGP.train_cm`, `GPTrainer.val_cm` | `BooleanGP.evaluator.confusion_matrix(rule)`, `GPTrainer.val_evaluator.confusion_matrix(rule)` |
| `GPTrainer.score_fn`, `GPTrainer.val_score_fn` | `GPTrainer.gp_algo.evaluator.score_fn` (validation uses the same function) |
| `GPTrainer.val_data`, `GPTrainer.val_labels` | `GPTrainer.val_evaluator.dataset` |
| Custom `Rule` subclasses with their own `evaluate` | Not supported by backends, which evaluate `Literal`, `And` and `Or` and raise a `TypeError` for other rule types |

`BooleanGP` gained a `dataset` argument, which child populations receive from their parent.
`BooleanGP.evaluate_population(data, labels, score_fn)` and `BooleanGP.evaluate_best(data, labels, score_fn=None)` keep their signatures; they score through the backend.

## Deprecation timeline

These work in 2.x with a warning and will be removed in a future 2.x release:

| Deprecated | Replacement | Warning |
|---|---|---|
| `Rule.evaluate(data)` | `hgp_lib.evaluation.predict(rule, data)` | `DeprecationWarning` |
| `hgp_lib.metrics` | `hgp_lib.results` | `DeprecationWarning` |
| `hgp_lib.utils.metrics` | `hgp_lib.evaluation` | `DeprecationWarning` |
| `hgp_lib.utils.ComplexityCheck` | `hgp_lib.rules.ComplexityCheck` | `DeprecationWarning` |
| `HGP_LOW_MEMORY` (ignored) | `NumpyBackend(evaluation=...)` | `FutureWarning` |
