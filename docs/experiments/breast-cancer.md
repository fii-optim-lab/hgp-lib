# Breast Cancer Wisconsin

This page compares the rules that HGP, a decision tree (DT), RIPPER, and BoolXAI learn on the Breast Cancer Wisconsin (Diagnostic) dataset.[^bcw]
The dataset ships with scikit-learn and has 569 tumors described by 30 numeric features: 357 benign and 212 malignant.

All methods use one stratified 80/20 train/test split with seed 42, and the positive class is benign.
Each method runs with its default parameters:

- HGP: `BooleanRuleClassifier` with 1000 epochs, on the bins of the default `StandardBinarizer`.
- DT: the scikit-learn decision tree.
- RIPPER: wittgenstein on the raw features, which it splits into its own bins.
- BoolXAI: on 5 quantile bins per feature.

The test set has 114 tumors, so a single split says little about accuracy.
See [PMLB](pmlb.md#results-on-pmlb) for the comparison over 70 datasets.

## Rules

Each rule below is true for the tumors it predicts benign.
The rules use the hgp-lib notation: `,` separates the operands of an operator, and `~` negates the literal or operator that follows.
The DT and RIPPER models are written as hgp-lib rules.
DT thresholds and BoolXAI bin edges are shown with 4 significant digits.

### HGP

Rule size: 21 nodes. Literals: 15. Test F1: 0.9437.

HGP evolves one rule of nested `And` and `Or` operators over the binarizer's bins.

```text
And(
  ~115.35 <= worst perimeter,
  And(
    ~0.125 <= mean compactness < 0.1293,
    ~1.807 <= texture error < 1.9,
    Or(
      Or(
        ~Or(
          16.8 <= worst radius < 17.51,
          696.25 <= mean area
        ),
        mean texture < 16.395
      ),
      29.3 <= worst texture < 30.91
    ),
    ~739.2 <= worst area < 743.45,
    ~0.176 <= worst concave points
  ),
  ~17.51 <= worst radius,
  ~17.51 <= worst radius,
  ~1.807 <= texture error < 1.9,
  ~739.2 <= worst area < 743.45,
  Or(
    ~mean texture < 16.395,
    ~1.807 <= texture error < 1.9
  )
)
```

A tumor is benign if its worst perimeter, worst radius, and worst concave points are below a threshold, its mean compactness, texture error, and worst area lie outside a bin, and one operand of the first `Or` holds.
hgp-lib does not simplify rules: three literals appear more than once, and the last `Or` always holds, because the rule already requires `~1.807 <= texture error < 1.9`.
Without them, the rule has 10 literals and 13 nodes and makes the same predictions.

### DT

Rule size: 56 nodes. Literals: 46. Test F1: 0.9286.

The tree becomes the disjunction of its 9 paths to benign leaves.

```text
Or(
  And(
    ~worst radius <= 16.8,
    ~texture error <= 0.4731,
    worst concavity <= 0.1907,
    worst texture <= 30.98
  ),
  And(
    ~worst radius <= 16.8,
    texture error <= 0.4731
  ),
  And(
    worst radius <= 16.8,
    ~worst concave points <= 0.1359,
    ~worst texture <= 25.62,
    worst symmetry <= 0.2682,
    mean concave points <= 0.06432
  ),
  And(
    worst radius <= 16.8,
    ~worst concave points <= 0.1359,
    worst texture <= 25.62,
    worst area <= 817.1,
    worst smoothness <= 0.1786
  ),
  And(
    worst radius <= 16.8,
    worst concave points <= 0.1359,
    area error <= 91.56,
    ~area error <= 38.6,
    ~texture error <= 2.007,
    concave points error <= 0.01308
  ),
  And(
    worst radius <= 16.8,
    worst concave points <= 0.1359,
    area error <= 91.56,
    ~area error <= 38.6,
    texture error <= 2.007
  ),
  And(
    worst radius <= 16.8,
    worst concave points <= 0.1359,
    area error <= 91.56,
    area error <= 38.6,
    ~smoothness error <= 0.003294,
    ~worst texture <= 33.27,
    ~worst texture <= 33.8
  ),
  And(
    worst radius <= 16.8,
    worst concave points <= 0.1359,
    area error <= 91.56,
    area error <= 38.6,
    ~smoothness error <= 0.003294,
    worst texture <= 33.27
  ),
  And(
    worst radius <= 16.8,
    worst concave points <= 0.1359,
    area error <= 91.56,
    area error <= 38.6,
    smoothness error <= 0.003294,
    worst texture <= 26.64
  )
)
```

Each path repeats the tests it shares with other paths, so the root test on worst radius appears in all 9 conjunctions.

### RIPPER

Rule size: 19 nodes. Literals: 15. Test F1: 0.8702.

RIPPER learns a list of rules for the positive class and predicts it when any rule fires.
As a single rule, this is a disjunction of conjunctions, here of 12 conjunctions with one or two literals each.
The bins are RIPPER's own, as wittgenstein prints them.

```text
Or(
  perimeter error = <1.28,
  mean concave points = 0.011 - 0.018,
  worst concavity = 0.089 - 0.14,
  concavity error = <0.007,
  worst perimeter = 86.17 - 90.53,
  worst radius = 11.16 - 12.47,
  worst radius = 13.85 - 14.91,
  And(
    worst concave points = 0.083 - 0.097,
    mean concave points = 0.033 - 0.046
  ),
  worst radius = <11.16,
  worst radius = 12.47 - 13.3,
  And(
    worst concavity = 0.14 - 0.18,
    mean texture = <14.0
  ),
  And(
    mean texture = 15.49 - 16.76,
    mean fractal dimension = 0.068 - 0.072
  )
)
```

Each literal tests one bin, so a range that spans several bins needs one conjunction per bin.
For example, three conjunctions cover worst radius below 13.3.

### BoolXAI

Rule size: 6 nodes. Literals: 5. Test F1: 0.8955.

BoolXAI refines one rule by simulated annealing.
Besides `And` and `Or`, it uses the higher-order Boolean operators `AtLeast`, `AtMost`, and `Choose`.
`AtMost1` is true when at most one of its operands is true.

```text
AtMost1(
  ~worst concavity < 0.08854,
  0.1524 <= mean concavity,
  15.85 <= worst radius < 20.13,
  1243 <= worst area,
  900.3 <= mean area
)
```

A tumor is benign if at most one of the five conditions holds, such as a worst area of at least 1243.

[^bcw]: Breast Cancer Wisconsin (Diagnostic), <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>, loaded with `sklearn.datasets.load_breast_cancer`.
