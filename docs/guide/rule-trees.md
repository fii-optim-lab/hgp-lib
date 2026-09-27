# Rule Trees

A rule is a tree of nodes.
Operators ([`And`](../api/rules.md#hgp_lib.rules.operators.And), [`Or`](../api/rules.md#hgp_lib.rules.operators.Or)) combine subrules, and literals ([`Literal`](../api/rules.md#hgp_lib.rules.literals.Literal)) test a single feature.
Any node can be negated.

```python
from hgp_lib.rules import And, Or, Literal

rule = And([
    Literal(value=0),
    Or([Literal(value=1, negated=True), Literal(value=2)]),
    Literal(value=3),
])
# And(0, Or(~1, 2), 3)
```

Rules only describe the logical structure.
Evaluating them against data is the job of an evaluation backend, see [Evaluating rules](#evaluating-rules).

The `rules` module is on the hot path.
Every candidate rule is evaluated against the data on every epoch, so the module is built for speed rather than safety.
This page describes the choices that make it fast.

## No runtime validation

Nodes do not check their inputs.
Operators assume they hold valid subrules, literals assume a valid feature index, and shapes are never checked.
The genetic operators that build rules are responsible for keeping them well-formed.
This removes per-node branching and lets evaluation stay tight.

## Slotted nodes

[`Rule`](../api/rules.md#hgp_lib.rules.rules.Rule) defines `__slots__` for its four fields (`subrules`, `parent`, `value`, `negated`).
Slots drop the per-instance `__dict__`, which lowers memory per node and speeds up attribute access.
With populations of many rules, each holding many nodes, this adds up.

## Evaluating rules

[`predict`](../api/evaluation.md#hgp_lib.evaluation.api.predict) evaluates a rule on binarized data: a 2D boolean array, with instances on rows and features on columns.
[`score`](../api/evaluation.md#hgp_lib.evaluation.api.score) also scores the predictions.

```python
from hgp_lib.evaluation import predict, score

y_pred = predict(rule, X_bin)       # one boolean prediction per row
f1 = score(rule, X_bin, y)          # fast_f1_score by default
```

For raw data, use [`BooleanRuleClassifier.predict`](../api/trainers.md#hgp_lib.trainers.boolean_rule_classifier.BooleanRuleClassifier) or [`GPBenchmarker.predict`](../api/benchmarkers.md#hgp_lib.benchmarkers.gp_benchmarker.GPBenchmarker), which binarize it with their fitted binarizer first.

A literal indexes its column once and returns the boolean vector for all instances at once.
Operators combine these vectors with NumPy boolean operations, so a whole rule resolves without Python-level loops over instances.

## The NumPy backend

[`NumpyBackend`](../api/evaluation.md#hgp_lib.evaluation.numpy.backend.NumpyBackend) is the default backend.
Its options only change speed and memory use, never results.

```python
from hgp_lib import BooleanGPConfig
from hgp_lib.evaluation import NumpyBackend

BooleanGPConfig(backend=NumpyBackend(order="F", low_memory=True, batched=False))  # the defaults
BooleanGPConfig(backend=NumpyBackend(batched=True, batch_size=50))  # 50 rules per batch
```

- `order` is the memory layout of the training data.
  Rules read feature columns, which are contiguous in `"F"` (column-major) order.
- `low_memory` chooses between two algorithms with the same results.
  With `True`, each operator updates one result buffer in place, and literal children are folded in by ufuncs that write into that buffer, so they allocate nothing.
  With `False`, the literal children of an operator are gathered with one fancy index into a block, then reduced with `all` or `any`.
  That needs fewer NumPy calls but a temporary block per operator.
- `batched` stacks the predictions of several rules into one boolean block, one row per rule, and scores the block with one call instead of one call per rule.
  `batch_size` is the number of rules per batch; `None` (the default) scores the whole population in one batch.
  A batch holds `batch_size` times the number of training rows booleans, so a smaller `batch_size` bounds the memory on large data.
  Only the built-in scorers are batched; a custom scorer is still called once per rule.

In the repository's backend benchmarks, `order="F"` with `low_memory=True` was the fastest in every scenario.
`batched` was faster on a few hundred to a few thousand training rows and slower on larger data, so it is off by default.
See [`NumpyBackend`](../api/evaluation.md#hgp_lib.evaluation.numpy.backend.NumpyBackend) for the measured speedups.

## The PyTorch backend

[`TorchBackend`](../api/evaluation.md#hgp_lib.evaluation.torch.backend.TorchBackend) evaluates rules with PyTorch, on the CPU or on a GPU.
It needs PyTorch, which is an optional dependency:

```bash
pip install "hgp-lib[torch]"
```

```python
from hgp_lib import BooleanGPConfig
from hgp_lib.evaluation import TorchBackend

BooleanGPConfig(backend=TorchBackend())                # on the CPU
BooleanGPConfig(backend=TorchBackend(device="cuda"))   # on the default CUDA GPU
BooleanGPConfig(backend=TorchBackend(device="mps", batched=True, batch_size=50))
```

- `device` is where the training rows are stored and rules are evaluated. `None` (the default) means `"cpu"`.
- `batched` and `batch_size` work as for the NumPy backend: the predictions of `batch_size` rules are stacked on the device and counted together, and `None` counts the whole population at once.

It gives the same scores, and so the same rules, as the NumPy backend.
For the built-in scorers, true positives and predicted positives are counted as exact integers on the device, and only these counts are copied back, once per population.
A custom scorer receives each prediction as a NumPy array, so every prediction is copied from the device.

Each literal of a rule is one tensor operation, and every operation has a fixed launch cost.
A GPU therefore only pays off on large data.
On an Apple M3 GPU, scoring 100 random rules was 1.6x faster than the NumPy backend on 1 million rows, and 4.6x slower on 100 thousand rows.
On the repository's benchmark datasets the NumPy backend was faster, so on the CPU use the NumPy backend.

With a GPU, [`GPBenchmarker`](../api/benchmarkers.md#hgp_lib.benchmarkers.gp_benchmarker.GPBenchmarker) runs each of its `n_jobs` worker processes on the same device, and each one holds its own copy of the training rows.
Keep `n_jobs` small enough for the data to fit in GPU memory.

## Scoring during training

During training, the backend binds the scorer to the training rows once, and scores every rule of every generation with it.
Binding computes constants such as the number of positive labels once, and picks the kernel for the dataset (for example the case without positive labels) once, not per rule.
The built-in scorers, [`fast_f1_score`](../api/evaluation.md#hgp_lib.evaluation.scorers.fast_f1_score) and [`fast_accuracy_score`](../api/evaluation.md#hgp_lib.evaluation.scorers.fast_accuracy_score), get dedicated kernels in both backends.

When `optimize_scorer` allows it, duplicate rows are merged into integer sample weights before binding.
The rules are then evaluated on fewer rows, and the counts are exact, so the scores are the same as on the original rows.
