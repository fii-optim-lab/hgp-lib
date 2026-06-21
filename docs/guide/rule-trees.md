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

## Vectorized evaluation

`evaluate` works on a 2D array, with instances on rows and features on columns.
A literal indexes its column once and returns the boolean vector for all instances at once.
Operators combine these vectors with NumPy boolean operations, so a whole rule resolves without Python-level loops over instances.

## Batched literal evaluation

The default [`And`](../api/rules.md#hgp_lib.rules.operators.And) and [`Or`](../api/rules.md#hgp_lib.rules.operators.Or) separate literal children from operator children before combining them.
All literal columns are gathered in a single fancy-index into the data, then reduced with `all` or `any` along one axis.
Negation is applied to the gathered block with a precomputed mask using XOR.
This evaluates the literal part of a node in one pass instead of one operation per literal, which is the common case for shallow rules.

```python
# inside And.evaluate, conceptually:
mask = (data[:, cols] ^ neg_mask).all(1)  # all literals at once
for op in sub_operators:                  # then fold in nested operators
    mask &= op.evaluate(data)
```

[`Or`](../api/rules.md#hgp_lib.rules.operators.Or) mirrors this with `any` and `|=`.
Negation of the whole node is done in place with `np.logical_not(mask, out=mask)` to avoid a fresh allocation.

## Two operator implementations

There are two operator backends with the same behavior but different trade-offs.

- `operators.py` is the default.
  It does as few operations as possible, at the cost of more temporary memory.
- `low_memory_operators.py` allocates as little as possible.
  It updates a single result buffer in place, copying only when the first child would alias the input data.

Select the low-memory backend with an environment variable before importing the package.

```bash
export HGP_LOW_MEMORY=1
```

The `hgp_lib.rules` package reads `HGP_LOW_MEMORY` at import time and binds [`And`](../api/rules.md#hgp_lib.rules.operators.And) and [`Or`](../api/rules.md#hgp_lib.rules.operators.Or) to the chosen backend.

## Traversal helpers

`flatten` collects all nodes iteratively with a queue, which avoids Python recursion overhead and stack limits on deep trees.
A recursive variant (`r_flatten`) is also available when a left-to-right preorder is needed.
`__len__` counts nodes with the same iterative approach.
