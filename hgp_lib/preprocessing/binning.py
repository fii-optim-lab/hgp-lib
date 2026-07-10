from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import TREE_UNDEFINED


class BinningStrategy(ABC):
    """
    Strategy for computing bin edges of a single numeric feature.

    A strategy turns a 1-D array of values (and optional labels) into a sorted array
    of bin edges. The edges always start with ``-inf`` and end with ``inf`` so that
    values outside the fitted range still fall into a bin. ``StandardBinarizer``
    delegates numeric binning to a strategy, so a custom binning method only needs to
    subclass this class and implement ``compute_edges``.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.preprocessing.binning import QuantileBinning
        >>> QuantileBinning().compute_edges(np.array([1.0, 2.0, 3.0, 4.0]), None, 2).tolist()
        [-inf, 2.5, inf]
    """

    @abstractmethod
    def compute_edges(
        self, values: np.ndarray, y: Optional[np.ndarray], n_bins: int
    ) -> np.ndarray:
        """
        Compute sorted bin edges for a single numeric feature.

        Args:
            values (np.ndarray):
                1-D array of feature values, with missing values already removed.
            y (np.ndarray | None):
                Optional 1-D label array aligned with ``values``, for supervised strategies.
            n_bins (int):
                Desired maximum number of bins.

        Returns:
            np.ndarray: Sorted bin edges beginning with ``-inf`` and ending with ``inf``.
        """
        raise NotImplementedError()


class QuantileBinning(BinningStrategy):
    """
    Unsupervised binning that places edges at evenly spaced quantiles.

    Each bin holds a similar number of samples. Duplicate edges are removed, so the
    actual number of bins may be fewer than ``n_bins`` when many values are identical.
    A feature with one or fewer unique values yields a single ``[-inf, inf]`` bin.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.preprocessing.binning import QuantileBinning
        >>> QuantileBinning().compute_edges(np.array([5.0, 5.0, 5.0]), None, 3).tolist()
        [-inf, inf]
    """

    def compute_edges(
        self, values: np.ndarray, y: Optional[np.ndarray], n_bins: int
    ) -> np.ndarray:
        if len(np.unique(values)) <= 1:
            return np.array([-np.inf, np.inf])

        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(values, quantiles)
        edges[0] = -np.inf
        edges[-1] = np.inf
        return np.unique(edges)


class SupervisedTreeBinning(BinningStrategy):
    """
    Supervised binning that uses a decision tree to place class-aware edges.

    A shallow decision tree is fit to predict ``y`` from the single feature, and its
    split thresholds become the bin edges. The edges maximize class separation, which
    typically produces more informative features than unsupervised binning. A feature
    with one or fewer unique values yields a single ``[-inf, inf]`` bin.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.preprocessing.binning import SupervisedTreeBinning
        >>> SupervisedTreeBinning().compute_edges(
        ...     np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1]), 2
        ... ).tolist()
        [-inf, 2.5, inf]
    """

    def compute_edges(
        self, values: np.ndarray, y: Optional[np.ndarray], n_bins: int
    ) -> np.ndarray:
        if y is None:
            raise ValueError("SupervisedTreeBinning requires labels y")
        if len(np.unique(values)) <= 1:
            return np.array([-np.inf, np.inf])

        tree = DecisionTreeClassifier(max_leaf_nodes=n_bins)
        tree.fit(values.reshape(-1, 1), y, check_input=False)
        thresholds = np.sort(tree.tree_.threshold[tree.tree_.feature != TREE_UNDEFINED])

        return np.concatenate([[-np.inf], thresholds, [np.inf]])
