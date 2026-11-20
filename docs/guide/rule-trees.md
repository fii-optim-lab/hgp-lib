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
Its default values were chosen based on benchmarking; see [Performance](../performance.md).

```python
from hgp_lib import BooleanGPConfig
from hgp_lib.evaluation import NumpyBackend

BooleanGPConfig(backend=NumpyBackend(evaluation="block"))
BooleanGPConfig(backend=NumpyBackend(batched=True, batch_size=50))  # 50 rules per batch
BooleanGPConfig(backend=NumpyBackend(dtype="float64"))  # float64 scores
```

- `order` is the memory layout of the training data.
  Rules read feature columns, which are contiguous in `"F"` (column-major) order.
- `evaluation` chooses between three implementations:
  - `"block"` gathers the literal children of an operator into one block with a single fancy index, applies their negations with one XOR, and reduces the block with `all` or `any`.
    It needs few NumPy calls, and a temporary block per operator.
  - `"inplace"` keeps one result buffer per operator and updates it in place with each child.
    A negated literal child is negated into a temporary first.
  - `"fused"` is `"inplace"` with the negation fused into the update.
    For booleans, `a & ~b` is `a > b` and `a | ~b` is `a >= b`, so a negated literal child takes one comparison ufunc that writes into the buffer.
- `batched` stacks the predictions of several rules into one boolean block, one row per rule, and scores the block with one call instead of one call per rule.
  `batch_size` is the number of rules per batch; `None` (the default) scores the whole population in one batch.
  A batch holds `batch_size` times the number of training rows booleans, so a smaller `batch_size` bounds the memory on large data.
  The built-in scorers are batched, and so are scoring functions with a `batched` method (see [Custom scorers](#custom-scorers)).
- `dtype` is the float type of the scores, `float32` by default.
  Counts stay exact: when the training rows stand for more original rows than `dtype` counts exactly (16,777,216 for `float32`), weighted counts switch to `float64`.

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
- `evaluation`, `batched`, `batch_size` and `dtype` work as for the NumPy backend, with tensors on `device`.
  MPS has no `float64`.

The training rows are stored once, on `device`, as a [`TorchDataset`](../api/evaluation.md#hgp_lib.evaluation.torch.dataset.TorchDataset).
Predictions and scores stay tensors on `device` until they reach NumPy code: once per generation for the scores, and at the end of the public `predict` and `score` functions.
A custom scoring function receives tensors, so it must work with tensors.

With a GPU, [`GPBenchmarker`](../api/benchmarkers.md#hgp_lib.benchmarkers.gp_benchmarker.GPBenchmarker) runs each of its `n_jobs` worker processes on the same device, and each one holds its own copy of the training rows.
Keep `n_jobs` small enough for the data to fit in GPU memory.
On Linux, worker processes are forked by default before Python 3.14, and a forked process cannot use CUDA when its parent already did.

## Scoring during training

During training, the backend binds the scoring function to the training rows once, and scores every rule of every generation with it.
Binding computes constants such as the number of positive labels once, and picks the kernel for the dataset (for example the case without positive labels) once, not per rule.
The built-in scorers, [`fast_f1_score`](../api/evaluation.md#hgp_lib.evaluation.scorers.fast_f1_score) and [`fast_accuracy_score`](../api/evaluation.md#hgp_lib.evaluation.scorers.fast_accuracy_score), get dedicated kernels in both backends.

When `optimize_scorer` allows it, duplicate rows are merged into integer sample weights before binding.
The rules are then evaluated on fewer rows, and the counts are exact, so the scores are those of the original rows.

### Custom scorers

A scoring function takes `(y_true, y_pred)`, and `sample_weight` when rows are merged, and returns a finite float.
It never returns NaN: selection and best-rule tracking compare scores, and every comparison with NaN is false.
With [`TorchBackend`](../api/evaluation.md#hgp_lib.evaluation.torch.backend.TorchBackend), the arguments are tensors.

A scoring function can also have a `batched` method, which scores a 2-D stack of predictions, one row per rule.
Backends with `batched=True` call it once per batch instead of calling the function once per rule:

```python
import numpy as np


class Recall:
    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.batched(y_true, y_pred[None], sample_weight)[0]

    def batched(self, y_true, y_preds, sample_weight=None):
        weights = np.ones(len(y_true)) if sample_weight is None else sample_weight
        positives = weights @ y_true
        if positives == 0:
            return np.ones(len(y_preds))
        return (y_preds & y_true) @ weights / positives


BooleanGPConfig(score_fn=Recall(), backend=NumpyBackend(batched=True))
```

With `optimize_scorer=True`, both the function and its `batched` method must accept `sample_weight`; otherwise binding raises a `ValueError`.
