"""Sampling strategies for hierarchical child population generation.

This module provides abstract base classes and data structures for sampling
data and features when creating child populations in hierarchical GP.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy import ndarray

from ..evaluation import Dataset
from ..utils.validation import check_isinstance


@dataclass
class SamplingResult:
    """Result of a sampling operation: the sampled rows and the feature mapping.

    Attributes:
        dataset: The sampled rows, labels and sample weights. Weights are ``None`` unless
            the child's rows stand for several original rows.
        feature_mapping: Dictionary mapping child feature indices to parent feature indices.
            Direction: child_index -> parent_index.

            For example, if the sampled features are [3, 7, 12], then feature_mapping is:
                {0: 3, 1: 7, 2: 12}

            This mapping is used during crossover to translate rules evolved in the child's
            reduced feature space back to the parent's full feature space. When a child rule
            references feature 0, applying this mapping converts it to feature 3 in the
            parent's space.

            Set to None for instance-only sampling where all features are preserved.
    """

    dataset: Dataset
    feature_mapping: dict[int, int] | None


class SamplingStrategy(ABC):
    """Abstract base class for data sampling strategies.

    Sampling strategies define how to select subsets of data and/or features
    for child populations in hierarchical GP. The parent's rows may be merged into
    sample weights; instance counts and fractions always refer to original rows, and
    `Dataset.take` selects original rows without expanding the data.

    Attributes:
        feature_fraction (float): Fraction of features to sample per child.
            Default: `1.0`.
        sample_fraction (float): Fraction of instances to sample per child.
            Default: `1.0`.
        replace (bool): Whether to allow overlap between children.
            Default: `False`.
        MIN_FEATURES: Minimum number of features required in sampled result.
        MIN_INSTANCES: Minimum number of instances required in sampled result.
    """

    MIN_FEATURES = 2
    MIN_INSTANCES = 2

    def __init__(
        self,
        feature_fraction: float = 1.0,
        sample_fraction: float = 1.0,
        replace: bool = False,
    ):
        check_isinstance(feature_fraction, float)
        check_isinstance(sample_fraction, float)
        check_isinstance(replace, bool)

        if feature_fraction <= 0 or feature_fraction > 1:
            raise ValueError(
                f"feature_fraction must be in (0, 1], is {feature_fraction}"
            )
        if sample_fraction <= 0 or sample_fraction > 1:
            raise ValueError(f"sample_fraction must be in (0, 1], is {sample_fraction}")

        self.feature_fraction = feature_fraction
        self.sample_fraction = sample_fraction
        self.replace = replace

    def allocate_indices_to_children(
        self, k: int, n: int, num_children: int
    ) -> list[ndarray] | ndarray:
        """Allocate `k` indices out of `n` to each of `num_children` children.

        When `k >= n`, every child receives all `n` indices. When
        `replace=False` and `k * num_children <= n`, indices are partitioned
        without overlap. Otherwise each child receives an independent random
        sample of `k` unique indices (overlap between children is possible).

        Args:
            k (int): Number of indices each child receives.
            n (int): Total number of available indices.
            num_children (int): Number of children to allocate to.

        Returns:
            list[ndarray] | ndarray: One index array per child (the rows of a 2D array
                when the indices are partitioned), each with `min(k, n)` indices.
        """
        if k >= n:
            return [np.arange(n) for _ in range(num_children)]
        if not self.replace and k * num_children <= n:
            return np.random.permutation(n)[: k * num_children].reshape(num_children, k)
        return [np.random.choice(n, size=k, replace=False) for _ in range(num_children)]

    @staticmethod
    def create_sampling_result(
        dataset: Dataset,
        feature_indices: ndarray | None,
        instance_indices: ndarray | None,
    ) -> SamplingResult:
        """Select the given original rows and feature columns, and build the result.

        Args:
            dataset: The parent's rows.
            feature_indices: Feature columns to keep, or None to keep all features.
            instance_indices: Original rows to keep, in ``range(dataset.n_rows)``, or None
                to keep all rows.

        Returns:
            SamplingResult: The sampled rows, with `feature_mapping` set from
                `feature_indices` (None when no features were sampled).

        Examples:
            >>> import numpy as np
            >>> from hgp_lib.evaluation import Dataset
            >>> from hgp_lib.populations import SamplingStrategy
            >>> dataset = Dataset(np.eye(3, dtype=bool), np.array([1, 0, 1]), np.array([3, 1, 2]))
            >>> result = SamplingStrategy.create_sampling_result(
            ...     dataset, np.array([2, 0]), np.array([0, 1, 5])
            ... )
            >>> result.dataset.sample_weight.tolist(), result.feature_mapping
            ([2, 1], {0: 2, 1: 0})
        """
        if instance_indices is not None:
            dataset = dataset.take(instance_indices)
        feature_mapping = None
        if feature_indices is not None:
            feature_mapping = {i: int(idx) for i, idx in enumerate(feature_indices)}
            dataset = dataset.select_features(feature_indices)
        return SamplingResult(dataset=dataset, feature_mapping=feature_mapping)

    @abstractmethod
    def sample(self, dataset: Dataset, num_children: int) -> list[SamplingResult]:
        """Sample data and/or features for child populations.

        Args:
            dataset: The parent's training rows, possibly merged into sample weights.
            num_children: Number of child populations to create.

        Returns:
            List of SamplingResult, one per child (exactly `num_children` elements).
        """


class FeatureSamplingStrategy(SamplingStrategy):
    """Samples a subset of features from the training data.

    Each child population receives a subset of the parent's feature columns.
    The number of features per child is `ceil(num_features * feature_fraction)`.

    Overlap behavior (controlled by `replace` parameter):
        - `replace=False`: No overlap between children (partitioning) — each feature
          appears in at most one child population.
        - `replace=True`: Overlap allowed — features can appear in multiple children.

    When `feature_fraction=1.0`, all children receive all features regardless of
    `replace`.

    Within each child, features are always unique (no duplicates within a single child).

    Attributes:
        feature_fraction (float): Fraction of features per child. Default: `1.0`.
        replace (bool): Allow feature overlap between children. Default: `False`.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset
        >>> np.random.seed(42)
        >>> strategy = FeatureSamplingStrategy(feature_fraction=0.5)
        >>> data = np.random.rand(100, 10) > 0.5
        >>> labels = np.random.randint(0, 2, 100)
        >>> results = strategy.sample(Dataset(data, labels), num_children=3)
        >>> len(results)
        3
        >>> results[0].dataset.data.shape
        (100, 5)
        >>> results[0].feature_mapping.keys()
        dict_keys([0, 1, 2, 3, 4])
    """

    def __init__(self, feature_fraction: float = 1.0, replace: bool = False):
        super().__init__(feature_fraction=feature_fraction, replace=replace)

    def sample(self, dataset: Dataset, num_children: int) -> list[SamplingResult]:
        """Sample features for child populations.

        Args:
            dataset: The parent's training rows, possibly merged into sample weights.
            num_children: Number of child populations to create.

        Returns:
            List of SamplingResult, one per child, with sampled feature columns,
            all rows and their weights preserved, and feature_mapping set.
        """
        num_features = dataset.data.shape[1]
        features_per_child = ceil(num_features * self.feature_fraction)
        if features_per_child < self.MIN_FEATURES:
            raise ValueError(
                f"Cannot sample less than {self.MIN_FEATURES} features. "
                f"There are only {num_features} features and feature_fraction is {self.feature_fraction}!"
            )
        feature_allocation = self.allocate_indices_to_children(
            features_per_child, num_features, num_children
        )

        return [
            self.create_sampling_result(dataset, feature_indices, None)
            for feature_indices in feature_allocation
        ]


class InstanceSamplingStrategy(SamplingStrategy):
    """Samples a subset of instances from the training data.

    Each child population receives a subset of the parent's rows. All features
    are preserved. The number of instances per child is
    `ceil(num_instances * sample_fraction)`, counted in original rows.

    Overlap behavior (controlled by `replace` parameter):
        - `replace=False`: No overlap between children (partitioning).
        - `replace=True`: Overlap allowed.

    When `sample_fraction=1.0`, all children receive all instances regardless of
    `replace`.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset
        >>> np.random.seed(42)
        >>> strategy = InstanceSamplingStrategy(sample_fraction=0.8)
        >>> data = np.random.rand(100, 10) > 0.5
        >>> labels = np.random.randint(0, 2, 100)
        >>> results = strategy.sample(Dataset(data, labels), num_children=3)
        >>> len(results)
        3
        >>> results[0].dataset.data.shape
        (80, 10)
        >>> results[0].feature_mapping is None
        True
    """

    def __init__(self, sample_fraction: float = 1.0, replace: bool = False):
        super().__init__(sample_fraction=sample_fraction, replace=replace)

    def sample(self, dataset: Dataset, num_children: int) -> list[SamplingResult]:
        """Sample instances for child populations.

        Args:
            dataset: The parent's training rows, possibly merged into sample weights.
                Instances are sampled from the original rows the weights stand for.
            num_children: Number of child populations to create.

        Returns:
            List of SamplingResult, one per child, with sampled rows and their
            weights, all features preserved, and feature_mapping set to None.
        """
        num_instances = dataset.n_rows
        samples_per_child = ceil(num_instances * self.sample_fraction)
        if samples_per_child < self.MIN_INSTANCES:
            # ValueError: Cannot sample less than 2 instances. There are only 1 instances and sample_fraction is 0.39!
            raise ValueError(
                f"Cannot sample less than {self.MIN_INSTANCES} instances. "
                f"There are only {num_instances} instances and sample_fraction is {self.sample_fraction}!"
            )
        sample_allocation = self.allocate_indices_to_children(
            samples_per_child, num_instances, num_children
        )

        return [
            self.create_sampling_result(dataset, None, sample_indices)
            for sample_indices in sample_allocation
        ]


class CombinedSamplingStrategy(SamplingStrategy):
    """Combines feature and instance sampling.

    Applies both feature sampling and instance sampling to create
    child populations with reduced feature and instance sets.

    Attributes:
        feature_fraction (float): Fraction of features per child. Default: `1.0`.
        sample_fraction (float): Fraction of instances per child. Default: `1.0`.
        replace (bool): Whether to allow overlap between children. Default: `False`.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset
        >>> np.random.seed(42)
        >>> strategy = CombinedSamplingStrategy(
        ...     feature_fraction=0.5,
        ...     sample_fraction=0.5,
        ...     replace=False
        ... )
        >>> data = np.random.rand(100, 10) > 0.5
        >>> labels = np.random.randint(0, 2, 100)
        >>> results = strategy.sample(Dataset(data, labels), num_children=3)
        >>> len(results)
        3
        >>> results[0].dataset.data.shape
        (50, 5)
    """

    def __init__(
        self,
        feature_fraction: float = 1.0,
        sample_fraction: float = 1.0,
        replace: bool = False,
    ):
        super().__init__(
            feature_fraction=feature_fraction,
            sample_fraction=sample_fraction,
            replace=replace,
        )

    def sample(self, dataset: Dataset, num_children: int) -> list[SamplingResult]:
        """Sample both features and instances for all children at once.

        Args:
            dataset: The parent's training rows, possibly merged into sample weights.
                Instances are sampled from the original rows the weights stand for.
            num_children: Number of child populations to create.

        Returns:
            List of SamplingResult, one per child, with both feature and instance
            subsets applied, the weights of the sampled rows, and feature_mapping set.
        """
        num_instances = dataset.n_rows
        num_features = dataset.data.shape[1]
        samples_per_child = ceil(num_instances * self.sample_fraction)
        features_per_child = ceil(num_features * self.feature_fraction)
        if samples_per_child < self.MIN_INSTANCES:
            raise ValueError(
                f"Cannot sample less than {self.MIN_INSTANCES} instances. "
                f"There are only {num_instances} instances and sample_fraction is {self.sample_fraction}!"
            )
        if features_per_child < self.MIN_FEATURES:
            raise ValueError(
                f"Cannot sample less than {self.MIN_FEATURES} features. "
                f"There are only {num_features} features and feature_fraction is {self.feature_fraction}!"
            )
        sample_allocation = self.allocate_indices_to_children(
            samples_per_child, num_instances, num_children
        )
        feature_allocation = self.allocate_indices_to_children(
            features_per_child, num_features, num_children
        )

        return [
            self.create_sampling_result(dataset, feature_indices, sample_indices)
            for sample_indices, feature_indices in zip(
                sample_allocation, feature_allocation
            )
        ]
