# Data Preparation

Boolean GP operates on boolean data.
The [`StandardBinarizer`](../api/preprocessing.md#hgp_lib.preprocessing.binarizer.StandardBinarizer) converts numeric, categorical, and boolean columns into a purely boolean DataFrame.
Boolean columns are kept as is, categorical columns are one-hot encoded, and numeric columns are split into bins.

!!! note
    Binarization is label-aware.
    Numeric columns use class-aware bins computed from the training labels, which produces splits that separate the classes.
    To prevent data leakage, the binarizer is fit only on the training data, then applied to the validation and test data.
    See [Binarization](binarization.md) for how the binarizer works and its parameters.

## Manual binarization

For manual training with [`GPTrainer`](../api/trainers.md#hgp_lib.trainers.gp_trainer.GPTrainer) or [`BooleanGP`](../api/algorithms.md#hgp_lib.algorithms.boolean_gp.BooleanGP), binarize the data yourself.
Fit the binarizer on the training split and transform the rest.

```python
from hgp_lib.preprocessing import StandardBinarizer
from sklearn.model_selection import train_test_split

data, labels = ...  # pandas DataFrame + numpy array

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels,
)
train_data, val_data, train_labels, val_labels = train_test_split(
    train_data, train_labels, test_size=0.2, random_state=42, stratify=train_labels,
)

binarizer = StandardBinarizer(
    num_bins=5,  # applied only to numerical features
)
train_data = binarizer.fit_transform(train_data, train_labels)
val_data = binarizer.transform(val_data)
test_data = binarizer.transform(test_data)
```

Depending on usage, data preparation may vary.
For k-fold cross-validation, a fresh binarizer must be fit on each training fold.

## Automatic binarization

If you use [`GPBenchmarker`](../api/benchmarkers.md#hgp_lib.benchmarkers.gp_benchmarker.GPBenchmarker), you do not need to binarize manually.
The benchmarker fits a fresh binarizer on each training fold and applies it to the matching validation fold.
The best fold's binarizer is then used on the held-out test set.
This keeps binarization free of leakage across folds and splits.
See [Benchmarking](benchmarking.md).
