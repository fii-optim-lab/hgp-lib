# Binarization

Boolean GP works on boolean data.
The [`StandardBinarizer`](../api/preprocessing.md#hgp_lib.preprocessing.binarizer.StandardBinarizer) converts a mixed-type DataFrame into a purely boolean one.
Each output column is a boolean feature that a rule can test.

## How columns are converted

The binarizer handles three kinds of columns.

- Boolean columns pass through unchanged.
- Categorical columns are one-hot encoded into one boolean column per unique value.
- Numeric columns are split into bins, and each bin becomes one boolean column.

A numeric column with `k` bins produces up to `k` boolean columns.
A bin column is true when the original value falls inside that bin.

## Numeric binning strategies

The bin edges for a numeric column are learned at fit time.
The strategy depends on whether labels are passed to `fit_transform`.

When labels are provided, the binarizer uses a decision-tree strategy.
It trains a decision tree on the single feature against the labels and takes the split thresholds as bin edges.
The tree picks up to `k - 1` thresholds with the highest information gain, which produces `k` bins aligned with the class boundaries.
This is label-aware, so the bins separate the classes better than unsupervised binning.[@albert2025evolving]

When no labels are provided, the binarizer uses quantile binning.
Edges are placed at evenly spaced quantiles, so each bin holds a similar number of samples.
Quantile binning does not use the labels, so it can miss class-separating thresholds.

In both strategies, a column with one or fewer unique values produces a single bin.
The bins are open at the ends, from `-inf` to `+inf`, so values outside the training range still map to a bin.

The number of bins `k` is a trade-off.
A larger `k` captures finer decision boundaries, but it produces more boolean columns and a larger search space.
A smaller `k` keeps rules simple and training fast, at the cost of granularity.
Treat `k` as a hyperparameter to tune.

## Parameters

The [`StandardBinarizer`](../api/preprocessing.md#hgp_lib.preprocessing.binarizer.StandardBinarizer) exposes the following parameters.

- `num_bins`: the default number of bins for numeric columns. Must be at least 2. Default is 5.
- `column_strategy`: a per-column override for the number of bins, as a dict of column name to bin count. Default is none.
- `precision`: the number of decimals used when formatting numeric bin boundary names. Must be at least 0. Default is 3.

The `column_strategy` parameter is what sets this binarizer apart from the common scikit-learn discretizers.
A tool such as `KBinsDiscretizer` applies one bin count to all numeric columns.
Here you can give each column its own bin count, so a feature with a complex distribution can use more bins while a simple one uses fewer.

```python
from hgp_lib.preprocessing import StandardBinarizer

binarizer = StandardBinarizer(
    num_bins=5,                       # default for every numeric column
    column_strategy={"amount": 10},   # amount uses 10 bins instead
)
```

## Readable bin names

Each numeric bin column is named after its boundaries, so a rule that uses it reads in plain terms.

- A bin open on the left is named `column < right`.
- A bin open on the right is named `left <= column`.
- A bin with both edges finite is named `left <= column < right`.

These names flow into [`Rule.to_str(feature_names)`](../api/rules.md#hgp_lib.rules.rules.Rule), which is how a trained rule prints with feature names instead of indices.

## Fit once, transform many

Call `fit_transform` on the training data to learn the bin edges and categorical mappings.
Call `transform` on validation or test data to apply the same encoding.
The binarizer must be fit before `transform`, and the input must have the same columns, in the same order and dtypes, as the fitting data.

```python
binarizer = StandardBinarizer(num_bins=5)
train_bool = binarizer.fit_transform(train_data, train_labels)
test_bool = binarizer.transform(test_data)
```

Fitting only on the training data is what prevents leakage.
For benchmarking, the [`GPBenchmarker`](../api/benchmarkers.md#hgp_lib.benchmarkers.gp_benchmarker.GPBenchmarker) handles this per fold, so you do not call the binarizer yourself.
See [Benchmarking](benchmarking.md) for the workflow.

## References

\bibliography
