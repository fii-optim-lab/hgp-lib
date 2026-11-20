import warnings
from typing import NamedTuple

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from ..dataset import Dataset, take_weighted


def column_major(data: ndarray, device: torch.device) -> Tensor:
    """
    A 2-D NumPy array as a column-major tensor on ``device``.

    The tensor has the shape of ``data`` (rows, features), and each feature column is
    contiguous, like a NumPy array in ``"F"`` order. On the CPU it shares memory with
    ``data`` when ``data`` is already in ``"F"`` order; the library never writes to it.

    Args:
        data (ndarray): A 2-D array.
        device (torch.device): The device of the tensor.

    Returns:
        Tensor: The data on ``device``, with contiguous columns.
    """
    columns = np.asarray(data, order="F").T  # C-contiguous: one row per feature
    with warnings.catch_warnings():
        # Read-only arrays are shared as they are; the library only reads them.
        warnings.filterwarnings("ignore", "The given NumPy array is not writable")
        tensor = torch.from_numpy(columns)
    return tensor.to(device).T


def to_device(values: ndarray, device: torch.device) -> Tensor:
    """A 1-D NumPy array as a tensor on ``device``."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The given NumPy array is not writable")
        return torch.from_numpy(np.ascontiguousarray(values)).to(device)


class TorchDataset(NamedTuple):
    """
    The rows of a `TorchBackend`: the `Dataset` of PyTorch.

    Same attributes and methods as `Dataset`, with tensors on the backend's device.
    ``data`` is column-major: its shape is (rows, features) and each feature column is
    contiguous. Indices passed to `take` and `select_features` are NumPy arrays, as the
    population and sampling strategies draw them.

    Attributes:
        data (Tensor): 2-D boolean tensor, rows are instances and columns are features.
        labels (Tensor): 1-D boolean labels, one per row.
        sample_weight (Tensor | None): 1-D ``int64`` row counts, or ``None``.
            Default: `None`.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from hgp_lib.evaluation import Dataset
        >>> from hgp_lib.evaluation.torch import TorchDataset
        >>> data = np.array([[True, False], [True, False], [False, True]])
        >>> dataset = TorchDataset.from_numpy(Dataset(data, np.array([True, True, False])), torch.device("cpu"))
        >>> merged = dataset.deduplicate()
        >>> merged.sample_weight.tolist(), merged.n_rows, merged.n_features
        ([1, 2], 3, 2)
    """

    data: Tensor
    labels: Tensor
    sample_weight: Tensor | None = None

    @classmethod
    def from_numpy(cls, dataset: Dataset, device: torch.device) -> "TorchDataset":
        """Copy a NumPy `Dataset` to ``device`` (on the CPU, the data is shared)."""
        weights = dataset.sample_weight
        return cls(
            column_major(dataset.data, device),
            to_device(dataset.labels, device),
            None if weights is None else to_device(weights, device),
        )

    def to_numpy(self) -> Dataset:
        """Copy the rows to a NumPy `Dataset`."""
        weights = self.sample_weight
        return Dataset(
            self.data.cpu().numpy(),
            self.labels.cpu().numpy(),
            None if weights is None else weights.cpu().numpy(),
        )

    @property
    def n_rows(self) -> int:
        """Number of original rows: ``len(labels)`` without weights, their sum with weights."""
        if self.sample_weight is None:
            return len(self.labels)
        return int(self.sample_weight.sum())

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return self.data.shape[1]

    def deduplicate(self) -> "TorchDataset":
        """
        Merge rows with equal features and label, adding up their weights.

        The rows are merged on the host with `Dataset.deduplicate`, so a temporary host
        copy of the rows exists while merging. Returns the same dataset when there are no
        duplicates.

        Returns:
            TorchDataset: The merged dataset, or ``self``.
        """
        rows = self.to_numpy()
        merged = rows.deduplicate()
        if merged is rows:
            return self
        return TorchDataset.from_numpy(merged, self.data.device)

    def take(self, rows: ndarray) -> "TorchDataset":
        """
        Select original rows, by index in ``range(n_rows)``, as `Dataset.take`.

        Args:
            rows (ndarray): Distinct indices of original rows, in ``range(n_rows)``.

        Returns:
            TorchDataset: The selected rows.
        """
        device = self.data.device
        if self.sample_weight is None:
            index = to_device(rows, device)
            return TorchDataset(self.data.T[:, index].T, self.labels[index])
        selected, counts = take_weighted(self.sample_weight.cpu().numpy(), rows)
        index = to_device(selected, device)
        return TorchDataset(
            self.data.T[:, index].T,
            self.labels[index],
            None if counts is None else to_device(counts, device),
        )

    def select_features(self, columns: ndarray) -> "TorchDataset":
        """
        Keep only the given feature columns, as `Dataset.select_features`.

        Args:
            columns (ndarray): Indices of the columns to keep.

        Returns:
            TorchDataset: The dataset restricted to ``columns``.
        """
        index = to_device(columns, self.data.device)
        return TorchDataset(self.data.T[index].T, self.labels, self.sample_weight)
