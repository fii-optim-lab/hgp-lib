# Postprocessing

An evolved rule often says more than it needs to.
Genetic programming copies subtrees around, so the same literal can appear several times, and a part of the rule can be implied by another part.
The bins of a numeric column are separate literals, so a rule may say `Or(x < 1, 1 <= x < 2)` where `x < 2` would do.
[`hgp_lib.postprocessing`](../api/postprocessing.md) rewrites a rule into a smaller rule that makes the same predictions, and can write it over the original columns.

## Postprocessing a classifier's rule

A fitted [`BooleanRuleClassifier`](../api/trainers.md#hgp_lib.trainers.boolean_rule_classifier.BooleanRuleClassifier) postprocesses its rule with the features of its binarizer.

```python
from hgp_lib import BooleanRuleClassifier

clf = BooleanRuleClassifier()
clf.fit(X_train, y_train)

rule = clf.postprocess()                        # a Rule over the same features
text = clf.postprocess_to_str(indent=0)         # the same, over the original columns
dnf = clf.postprocess(canonical="dnf")          # an Or of And terms
cnf = clf.postprocess_to_str(canonical="cnf")   # an And of Or clauses, as a string
```

`postprocess` returns a [`Rule`](../api/rules.md#hgp_lib.rules.rules.Rule), so the result can be evaluated like the original one, for example with `predict(rule, clf.binarizer.transform(X_test).to_numpy())`.
The classifier itself keeps predicting with `clf.rule`.

The same functions work on any rule.
Pass the fitted binarizer as the features:

```python
from hgp_lib.postprocessing import postprocess, postprocess_to_str

rule = history.global_best_rule
smaller = postprocess(rule, binarizer)
print(postprocess_to_str(rule, binarizer))
```

Both functions take the same two flags:

- `optimize` (default `True`) simplifies the rule, and merges the literals that test the same original column.
- `canonical` (default `None`) is `None` to keep the structure of the rule, or `"dnf"` or `"cnf"` for a normal form.

## Example

The rule that HGP learns on the [Breast Cancer Wisconsin](../experiments/breast-cancer.md) dataset has 21 nodes and 15 literals.
Three literals appear twice, and its last `Or` always holds, because the rule already requires `~1.807 <= texture error < 1.9`.

`postprocess(rule, binarizer)` returns a rule of 13 nodes and 10 literals that makes the same predictions:

```text
And(
  ~115.35 <= worst perimeter,
  ~0.125 <= mean compactness < 0.1293,
  ~1.807 <= texture error < 1.9,
  Or(
    And(
      ~16.8 <= worst radius < 17.51,
      ~696.25 <= mean area
    ),
    mean texture < 16.395,
    29.3 <= worst texture < 30.91
  ),
  ~739.2 <= worst area < 743.45,
  ~0.176 <= worst concave points,
  ~17.51 <= worst radius
)
```

`postprocess_to_str(rule, binarizer, indent=0)` writes it over the original columns.
`~17.51 <= worst radius` becomes `worst radius < 17.51`, and inside the rule, which already requires it, `~16.8 <= worst radius < 17.51` becomes `worst radius < 16.8`:

```text
And(
  worst perimeter < 115.35,
  ~0.125 <= mean compactness < 0.1293,
  ~1.807 <= texture error < 1.9,
  Or(
    And(
      worst radius < 16.8,
      mean area < 696.25
    ),
    mean texture < 16.395,
    29.3 <= worst texture < 30.91
  ),
  ~739.2 <= worst area < 743.45,
  worst concave points < 0.176,
  worst radius < 17.51
)
```

## What optimizing does

With `optimize=True`, the rule is simplified until nothing changes:

- Negations are pushed down to the literals, and nested operators of the same kind are flattened: `And(a, And(b, c))` becomes `And(a, b, c)`.
- The literals on the same column inside an operator are merged. With the bins `x < 1`, `1 <= x < 2`, `2 <= x < 3` and `3 <= x`:
    - `And(x < 1, x < 1)` becomes `x < 1`;
    - `And(x < 1, 1 <= x < 2)` is always false, since bins exclude each other;
    - `And(~x < 1, ~1 <= x < 2, ~2 <= x < 3)` becomes `3 <= x`;
    - `Or(x < 1, 1 <= x < 2, 2 <= x < 3, 3 <= x)` is always true.
- A literal that the rest of the rule already decides is replaced by that constant. Inside an `And`, its other literals hold; inside an `Or`, they are false. So `And(a, Or(a, b))` becomes `a`, and `Or(a, And(~a, b))` becomes `Or(a, b)`.
- Constants, repeated parts and absorbed parts are removed: `And(a, Or(l, ~l))` becomes `a`, and `Or(A, And(A, B))` becomes `A`.

The optimized rule never has more nodes than the input.
A rule that is always true or always false is returned as `Or(l, ~l)` or `And(l, ~l)`, with `l` its first literal, because a rule has no constant node.
A rule of at most 3 nodes, such as `Or(l, ~l)` itself, is then returned unchanged.

## Rules and strings

A `Rule` can only use the existing features, so merging never creates a new bin edge.
In a rule, merging removes literals or whole operators.
With 5 bins on `x` (edges 1, 2, 3, 4), `Or(1 <= x < 2, 2 <= x < 3, 3 <= x < 4)` (4 nodes) becomes `And(~x < 1, ~4 <= x)` (3 nodes).

A string can say more.
[`postprocess_to_str`](../api/postprocessing.md#hgp_lib.postprocessing.api.postprocess_to_str) writes the literals of one column inside an operator as one condition, so the same rule becomes `1.0 <= x < 4.0`.
It uses the notation of `Rule.to_str`:

| Column | Conditions |
|---|---|
| numeric | `x < 2.5`, `1.5 <= x < 4.0`, `2.5 <= x`, and an `Or` of intervals for bins that are not adjacent |
| categorical | `color=red`, `color in {red, blue}` |
| boolean | `flag`, `~flag` |
| missing values | `x is NA`, `x is not NA` |
| constants | `True`, `False` |

Every comparison is false for a missing value; only `x is NA` matches it.
`~` is plain negation, so `~1.5 <= x < 4.0` is true for a missing value, and `Or(x < 1.5, 4.0 <= x)` is not.
Each condition is written in the shorter of its positive form and its negated form, so with the categories `red`, `green` and `blue`, `color in {red, green}` is written `~color=blue`.

With `optimize=False`, `postprocess_to_str` writes the literals with the feature names, like `rule.to_str(binarizer.get_feature_names_out())`.

## Canonical forms

`canonical="dnf"` returns a disjunctive normal form, an `Or` of `And` terms.
`canonical="cnf"` returns a conjunctive normal form, an `And` of `Or` clauses.
The value is case-insensitive, and [`CanonicalForm.DNF`](../api/postprocessing.md#hgp_lib.postprocessing.api.CanonicalForm) and `CanonicalForm.CNF` work too.

With `optimize=True`, the result is the canonical form of the rule:

1. The rule is expanded into terms that allow a set of values for each column.
2. All prime terms are computed: the terms that stop implying the rule as soon as they allow one more value.
3. The primes that the others already cover are removed, from the least to the most preferred (more conditions, then fewer allowed values).

The canonical form depends only on what the rule computes.
Two rules that make the same predictions on every row get the same canonical form, however they are written.
It is not always the smallest normal form, since finding that is NP-hard.
The CNF is computed the same way on the negated rule.
A rule that is always true or always false is `Or(0, ~0)` or `And(0, ~0)`, over feature 0, so that it is unique too.

In a `Rule`, a term that allows several bins of a column lists the excluded bins, to keep the shape of a DNF: `And(~x < 1, a)` rather than `And(Or(1 <= x < 2, 2 <= x), a)`.
The string writes it as one condition, `And(1.0 <= x, a)`.

With `optimize=False`, the rule is converted mechanically: negations are pushed down, operators are flattened, and `And` is distributed over `Or` (DNF) or `Or` over `And` (CNF).
Nothing is removed, so repeated literals and contradictory terms stay.

### Cost

Use `canonical` with caution on rules of more than 100 nodes.
A normal form can be exponentially larger than the rule, so computing it can take very long.
Evolved rules usually shrink a lot when they are optimized, which happens first, and rules that are already close to the requested form stay small.
Optimizing without a canonical form is fast at any size.

## Assumptions

The postprocessed rule makes the same predictions as the input on every row the binarizer can produce from data like the training data.
Two cases are excluded, and [`StandardBinarizer.transform`](../api/preprocessing.md#hgp_lib.preprocessing.binarizer.StandardBinarizer.transform) warns about both:

- A missing value in a column that had none at fit ([`UnseenNaNWarning`](../api/preprocessing.md#hgp_lib.preprocessing.warnings.UnseenNaNWarning)).
- A category that was not seen at fit ([`UnseenCategoryWarning`](../api/preprocessing.md#hgp_lib.preprocessing.warnings.UnseenCategoryWarning)).

The binarizer encodes such a row as all-false for that column.
On it, `Or(x < 2.5, 2.5 <= x)` is false, while postprocessing treats it as always true, so the postprocessed rule can predict the row differently.

## Features

Postprocessing needs to know which features come from the same original column: `And(x < 1, 1 <= x < 2)` is always false only because both are bins of `x`.
The features argument accepts:

- a fitted [`Binarizer`](../api/preprocessing.md#hgp_lib.preprocessing.base.Binarizer), whose [`get_feature_specs`](../api/preprocessing.md#hgp_lib.preprocessing.base.Binarizer.get_feature_specs) describes each feature;
- a list of [`FeatureSpec`](../api/preprocessing.md#hgp_lib.preprocessing.features.FeatureSpec) objects, the same descriptions;
- a list of feature names, read as independent boolean features;
- `None` (the default), which treats every literal as an independent boolean feature named by its index.

`StandardBinarizer` describes each feature with a `BooleanFeature`, `CategoryFeature`, `BinFeature` or `MissingFeature`, so nothing has to be parsed from the feature names.
Other binarizers, such as [`SklearnBinarizer`](../api/preprocessing.md#hgp_lib.preprocessing.sklearn_binarizer.SklearnBinarizer), describe each feature as an independent boolean.
The result is still exact, but literals of the same column are not merged.
