# Binarization

Boolean GP works on boolean data.
A [`Binarizer`](../api/preprocessing.md#hgp_lib.preprocessing.base.Binarizer) converts a mixed-type DataFrame into a purely boolean one.
Each output column is a boolean feature that a rule can test.
The default implementation is [`StandardBinarizer`](../api/preprocessing.md#hgp_lib.preprocessing.binarizer.StandardBinarizer).

## The Binarizer contract

Every binarizer follows the same contract, so implementations are interchangeable.

- `fit_transform(X, y=None)` learns the encoding and returns a boolean DataFrame.
- `transform(X)` applies the learned encoding and returns a boolean DataFrame with the same columns, in the same order.
- `is_fitted` reports whether the binarizer has been fitted.

[`GPBenchmarker`](../api/benchmarkers.md#hgp_lib.benchmarkers.gp_benchmarker.GPBenchmarker) relies on this contract, fitting a fresh copy per fold.

## How columns are converted

The `StandardBinarizer` handles three kinds of columns.

- Boolean columns pass through unchanged.
- Categorical, string, and object columns are one-hot encoded into one boolean column per unique value.
- Numeric columns are split into bins, and each bin becomes one boolean column.

A numeric column with `k` bins produces up to `k` boolean columns.
A bin column is true when the original value falls inside that bin.

String and object columns are treated as categorical, and this raises a [`StringColumnWarning`](../api/preprocessing.md#hgp_lib.preprocessing.warnings.StringColumnWarning).
Set the column to a pandas `category` dtype to make the intent explicit and avoid the warning.
A categorical column whose values are all distinct is skipped with a [`HighCardinalityWarning`](../api/preprocessing.md#hgp_lib.preprocessing.warnings.HighCardinalityWarning), since one-hot encoding it would produce one feature per row.

## Numeric binning strategies

Numeric bin edges are computed by a [`BinningStrategy`](../api/preprocessing.md#hgp_lib.preprocessing.binning.BinningStrategy).
When no strategy is set, the binarizer picks one based on whether labels are passed to `fit_transform`.

With labels, it uses [`SupervisedTreeBinning`](../api/preprocessing.md#hgp_lib.preprocessing.binning.SupervisedTreeBinning).
It trains a decision tree on the single feature against the labels and takes the split thresholds as bin edges.
The tree picks up to `k - 1` thresholds with the highest information gain, which produces `k` bins aligned with the class boundaries.
This is label-aware, so the bins separate the classes better than unsupervised binning.[@albert2025evolving]

Without labels, it uses [`QuantileBinning`](../api/preprocessing.md#hgp_lib.preprocessing.binning.QuantileBinning).
Edges are placed at evenly spaced quantiles, so each bin holds a similar number of samples.
Quantile binning does not use the labels, so it can miss class-separating thresholds.

In both strategies, a column with one or fewer unique values produces a single bin.
The bins are open at the ends, from `-inf` to `+inf`, so values outside the training range still map to a bin.

The number of bins `k` is a trade-off.
A larger `k` captures finer decision boundaries, but it produces more boolean columns and a larger search space.
A smaller `k` keeps rules simple and training fast, at the cost of granularity.
Treat `k` as a hyperparameter to tune.

## Parameters

The `StandardBinarizer` exposes the following parameters.

- `num_bins`: the default number of bins for numeric columns. Must be at least 2. Default is 5.
- `column_strategy`: a per-column override for the number of bins, as a dict of column name to bin count. Default is none.
- `precision`: the number of decimals used when formatting numeric bin boundary names. Must be at least 0. Default is 3.
- `numeric_binning`: an explicit [`BinningStrategy`](../api/preprocessing.md#hgp_lib.preprocessing.binning.BinningStrategy) for numeric columns. When set, it is used regardless of whether labels are provided. Default is none.

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

## Missing values

A column that contains missing values also produces a boolean `<col>_is_NA` indicator column, so a rule can test whether a value was missing.
At transform time, if a column has missing values that were not seen during fit, no new column is created and those rows are encoded as all-false, which raises a [`UnseenNaNWarning`](../api/preprocessing.md#hgp_lib.preprocessing.warnings.UnseenNaNWarning).

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

## Custom binning

To change only how numeric edges are chosen, implement a [`BinningStrategy`](../api/preprocessing.md#hgp_lib.preprocessing.binning.BinningStrategy) and pass it as `numeric_binning`.
The strategy receives the feature values, the optional labels, and the number of bins, and returns sorted edges bounded by `-inf` and `inf`.

```python
import numpy as np
from hgp_lib.preprocessing import StandardBinarizer
from hgp_lib.preprocessing.binning import BinningStrategy

class MedianSplit(BinningStrategy):
    def compute_edges(self, values, y, n_bins):
        return np.array([-np.inf, float(np.median(values)), np.inf])

binarizer = StandardBinarizer(numeric_binning=MedianSplit())
```

To change how a whole column type is handled, subclass `StandardBinarizer` and override the matching hooks: `_fit_boolean` / `_transform_boolean`, `_fit_categorical` / `_transform_categorical`, or `_fit_numeric` / `_transform_numeric`.
The `_fit_*` hooks return `(name, values)` pairs, and the `_transform_*` hooks return the values in the same order.

## Using a scikit-learn discretizer

To use a scikit-learn transformer instead, wrap it in [`SklearnBinarizer`](../api/preprocessing.md#hgp_lib.preprocessing.sklearn_binarizer.SklearnBinarizer).
The wrapper fits the transformer, converts its output to a boolean DataFrame, and derives readable column names.

```python
from sklearn.preprocessing import KBinsDiscretizer
from hgp_lib.preprocessing import SklearnBinarizer

binarizer = SklearnBinarizer(
    KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile")
)
```

Pass the wrapper anywhere a binarizer is accepted, including `BenchmarkerConfig.binarizer`.

## References

\bibliography
