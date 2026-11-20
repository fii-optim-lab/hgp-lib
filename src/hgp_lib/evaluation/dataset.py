from typing import NamedTuple

import numpy as np
from numpy import ndarray


def take_weighted(
    sample_weight: ndarray, rows: ndarray
) -> tuple[ndarray, ndarray | None]:
    """
    Which merged rows hold the original ``rows``, and how many of them each one holds.

    Merged row ``i`` stands for ``sample_weight[i]`` consecutive original rows.

    Args:
        sample_weight (ndarray): 1-D ``int64`` row counts of the merged rows.
        rows (ndarray): Distinct indices of original rows.

    Returns:
        tuple[ndarray, ndarray | None]: The indices of the selected merged rows, and
            their new counts, or ``None`` when every count is 1.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation.dataset import take_weighted
        >>> selected, counts = take_weighted(np.array([3, 1, 2]), np.array([0, 2, 4, 5]))
        >>> selected.tolist(), counts.tolist()
        ([0, 2], [2, 2])
    """
    owners = np.searchsorted(np.cumsum(sample_weight), rows, side="right")
    counts = np.bincount(owners, minlength=len(sample_weight))
    selected = np.flatnonzero(counts)
    counts = counts[selected]
    return selected, None if counts.max(initial=1) == 1 else counts


class Dataset(NamedTuple):
    """
    Binarized rows, their labels, and optionally how many original rows each one stands for.

    ``sample_weight`` is ``None`` when every row counts once. Weights only appear after
    `deduplicate` merges equal rows, and they are integer counts, so a weighted dataset
    gives the same scores as the original rows. Operations return ``None`` weights again
    whenever every count is 1.

    This is the NumPy dataset. Each backend has a dataset of its own with the same
    attributes and methods (``n_rows``, ``n_features``, ``deduplicate``, ``take`` and
    ``select_features``): `NumpyBackend` uses this class, and ``TorchBackend`` uses
    ``TorchDataset``. ``evaluator.dataset`` is always the backend's own dataset, so
    population and sampling strategies work on the backend's data without a copy.

    The data and labels are not checked: ``data`` must be boolean and ``labels`` must be
    boolean. The library checks user input once where it enters (for example in
    `BooleanGPConfig` or `hgp_lib.evaluation.score`).

    Attributes:
        data (ndarray): 2-D boolean array, rows are instances and columns are features.
        labels (ndarray): 1-D boolean labels, one per row.
        sample_weight (ndarray | None): 1-D ``int64`` row counts, or ``None``.
            Default: `None`.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset
        >>> data = np.array([[True, False], [True, False], [False, True]])
        >>> dataset = Dataset(data, np.array([True, True, False])).deduplicate()
        >>> dataset.sample_weight.tolist(), dataset.n_rows
        ([1, 2], 3)
    """

    data: ndarray
    labels: ndarray
    sample_weight: ndarray | None = None

    @property
    def n_rows(self) -> int:
        """
        Number of original rows: ``len(labels)`` without weights, their sum with weights.

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.evaluation import Dataset
            >>> Dataset(np.ones((2, 2), dtype=bool), np.array([True, False]), np.array([3, 4])).n_rows
            7
        """
        if self.sample_weight is None:
            return len(self.labels)
        return int(self.sample_weight.sum())

    @property
    def n_features(self) -> int:
        """
        Number of feature columns.

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.evaluation import Dataset
            >>> Dataset(np.ones((2, 3), dtype=bool), np.array([True, False])).n_features
            3
        """
        return self.data.shape[1]

    def deduplicate(self) -> "Dataset":
        """
        Merge rows with equal features and label, adding up their weights.

        Returns the same dataset when there are no duplicates. Unique rows are ordered by
        their packed bytes.

        Returns:
            Dataset: The merged dataset, with ``int64`` weights, or ``self``.

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.evaluation import Dataset
            >>> data = np.array([[True], [True], [False]])
            >>> merged = Dataset(data, np.array([True, True, False]), np.array([2, 3, 1])).deduplicate()
            >>> merged.data.ravel().tolist(), merged.sample_weight.tolist()
            ([False, True], [1, 5])
            >>> unique = Dataset(data[1:], np.array([True, False]))
            >>> unique.deduplicate() is unique
            True
        """
        # TODO: Benchmark hashing packed rows into uint64 and deduplicating the hashes,
        #  with an exact fallback on collisions.
        data, labels, sample_weight = self
        # Rows must be contiguous to be compared as raw bytes, whatever the data layout.
        rows = np.empty((len(labels), data.shape[1] + 1), dtype=bool)
        rows[:, :-1] = data
        rows[:, -1] = labels
        rows = np.packbits(rows, axis=1)
        row_view = rows.view(np.dtype((np.void, rows.shape[1]))).ravel()

        if sample_weight is None:
            _, unique_idx, counts = np.unique(
                row_view, return_index=True, return_counts=True
            )
        else:
            _, unique_idx, inverse = np.unique(
                row_view, return_index=True, return_inverse=True
            )
            counts = np.bincount(inverse.ravel(), weights=sample_weight)
        if len(unique_idx) == len(labels):
            return self
        return Dataset(data[unique_idx], labels[unique_idx], counts.astype(np.int64))

    def take(self, rows: ndarray) -> "Dataset":
        """
        Select original rows, by index in ``range(n_rows)``.

        With weights, merged row ``i`` stands for ``sample_weight[i]`` consecutive
        original rows, so drawing indices from ``range(n_rows)`` samples original rows
        without expanding the data. The result keeps one row per selected merged row,
        weighted by how many of its original rows were selected.

        Args:
            rows (ndarray): Distinct indices of original rows, in ``range(n_rows)``.

        Returns:
            Dataset: The selected rows.

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.evaluation import Dataset
            >>> dataset = Dataset(np.eye(3, dtype=bool), np.array([True, False, True]), np.array([3, 1, 2]))
            >>> subset = dataset.take(np.array([0, 2, 4, 5]))
            >>> subset.labels.tolist(), subset.sample_weight.tolist()
            ([True, True], [2, 2])
            >>> dataset.take(np.array([0, 3])).sample_weight is None
            True
        """
        if self.sample_weight is None:
            return Dataset(self.data[rows], self.labels[rows])
        selected, counts = take_weighted(self.sample_weight, rows)
        return Dataset(self.data[selected], self.labels[selected], counts)

    def select_features(self, columns: ndarray) -> "Dataset":
        """
        Keep only the given feature columns. Weights are kept; rows are not merged.

        Args:
            columns (ndarray): Indices of the columns to keep.

        Returns:
            Dataset: The dataset restricted to ``columns``.

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.evaluation import Dataset
            >>> dataset = Dataset(np.array([[True, False, True]]), np.array([True]))
            >>> dataset.select_features(np.array([2, 0])).data.tolist()
            [[True, True]]
        """
        return Dataset(self.data[:, columns], self.labels, self.sample_weight)
