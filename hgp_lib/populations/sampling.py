"""Sampling strategies for hierarchical child population generation.

This module provides abstract base classes and data structures for sampling
data and features when creating child populations in hierarchical GP.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil
from typing import Dict, List

import numpy as np
from numpy import ndarray

from hgp_lib.utils.validation import check_isinstance


# TODO: Add tests for this module


@dataclass
class SamplingResult:
    """Result of a sampling operation containing sampled data and index mappings.

    Attributes:
        data: Sampled training data as 2D boolean ndarray (instances x features).
        labels: Sampled labels as 1D integer ndarray.
        feature_indices: Selected feature indices from the parent's feature space as 1D ndarray.
            For example, if the parent has 20 features and we sample [3, 7, 12], the child's
            feature 0 corresponds to parent's feature 3, feature 1 to parent's 7, etc.
        instance_indices: Selected instance indices as 1D ndarray, or None for feature-only sampling.
        feature_mapping: Dictionary mapping child feature indices to parent feature indices.
            Direction: child_index -> parent_index.

            For example, if feature_indices is [3, 7, 12], then feature_mapping is:
                {0: 3, 1: 7, 2: 12}

            This mapping is used during crossover to translate rules evolved in the child's
            reduced feature space back to the parent's full feature space. When a child rule
            references feature 0, applying this mapping converts it to feature 3 in the
            parent's space.

            Set to None for instance-only sampling where all features are preserved.
    """

    data: ndarray
    labels: ndarray
    feature_indices: ndarray
    instance_indices: ndarray | None
    feature_mapping: Dict[int, int] | None


class SamplingStrategy(ABC):
    """Abstract base class for data sampling strategies.

    Sampling strategies define how to select subsets of data and/or features
    for child populations in hierarchical GP.

    Args:
        # TODO: Write documentation about feature_fraction, sample_fraction, and replace

    Attributes:
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

        self.add_feature_mapping = feature_fraction != 1.0

    def allocate_indices_to_children(self, k: int, n: int, num_children: int):
        # TODO: Write documentation
        if self.replace:
            return [
                np.random.choice(n, size=k, replace=False) for _ in range(num_children)
            ]
        if k * num_children > n:
            raise RuntimeError(
                f"Can't allocate {k} indices to {num_children} children from total {n} when replace is False"
            )
        return np.random.permutation(n)[: k * num_children].reshape(num_children, k)

    @staticmethod
    def create_sampling_result(data, labels, feature_indices, instance_indices):
        if instance_indices is not None:
            data = data[instance_indices]
            labels = labels[instance_indices]
        feature_mapping = None
        if feature_indices is not None:
            feature_mapping = {i: int(idx) for i, idx in enumerate(feature_indices)}
            data = data[:feature_indices]

        return SamplingResult(
            data=data,
            labels=labels,
            feature_mapping=feature_mapping,
            # TODO: Do we really need feature indices and instance indices here
            feature_indices=feature_indices,
            instance_indices=instance_indices,
        )

    @abstractmethod
    def sample(
        self,
        data: ndarray,
        labels: ndarray,
        num_children: int,
    ) -> List[SamplingResult]:
        """Sample data and/or features for child populations.

        Args:
            data: Training data as 2D boolean array (instances x features).
            labels: Training labels as 1D integer array.
            num_children: Number of child populations to create.

        Returns:
            List of SamplingResult, one per child (exactly `num_children` elements).
        """
        pass


class FeatureSamplingStrategy(SamplingStrategy):
    """Samples a subset of features from the training data.

    # TODO: Write documentation

    Overlap behavior (controlled by `replace` parameter):
        - replace=False: No overlap between children (partitioning) - each feature
          appears in at most one child population
        - replace=True: Overlap allowed - features can appear in multiple children

    Within each child, features are always unique (no duplicates within a single child).

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> strategy = FeatureSamplingStrategy(feature_fraction=0.5)
        >>> data = np.random.rand(100, 10) > 0.5
        >>> labels = np.random.randint(0, 2, 100)
        >>> results = strategy.sample(data, labels, num_features=10, num_children=3)
        >>> len(results)
        3
        >>> len(results[0].feature_indices)
        5
    """

    def __init__(self, feature_fraction: float = 1.0, replace: bool = False):
        super().__init__(feature_fraction, replace)

    def sample(
        self,
        data: ndarray,
        labels: ndarray,
        num_children: int,
    ) -> List[SamplingResult]:
        """Sample features for child populations.

        Args:
            data: Training data as 2D boolean array (instances x features).
            labels: Training labels as 1D integer array.
            num_children: Number of child populations to create.

        Returns:
            List of SamplingResult, one per child, with sampled feature columns,
            all instances preserved, and instance_indices set to None.
        """
        num_features = data.shape[1]
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
            self.create_sampling_result(data, labels, feature_indices, None)
            for feature_indices in feature_allocation
        ]


class InstanceSamplingStrategy(SamplingStrategy):
    """Samples a subset of instances from the training data.

    # TODO: Write documentation

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> strategy = InstanceSamplingStrategy(instance_fraction=0.8)
        >>> data = np.random.rand(100, 10) > 0.5
        >>> labels = np.random.randint(0, 2, 100)
        >>> results = strategy.sample(data, labels, num_features=10, num_children=3)
        >>> len(results)
        3
        >>> len(results[0].instance_indices)
        80
    """

    def __init__(self, sample_fraction: float = 1.0, replace: bool = False):
        super().__init__(sample_fraction=sample_fraction, replace=replace)

    def sample(
        self,
        data: ndarray,
        labels: ndarray,
        num_children: int,
    ) -> List[SamplingResult]:
        """Sample instances for child populations.

        Args:
            data: Training data as 2D boolean array (instances x features).
            labels: Training labels as 1D integer array.
            num_children: Number of child populations to create.

        Returns:
            List of SamplingResult, one per child, with sampled instance rows,
            all features preserved, and feature_mapping set to None.
        """
        num_instances = len(data)
        samples_per_child = ceil(num_instances * self.sample_fraction)
        if samples_per_child < self.MIN_INSTANCES:
            raise ValueError(
                f"Cannot sample less than {self.MIN_INSTANCES} instances. "
                f"There are only {num_instances} instances and sample_fraction is {self.sample_fraction}!"
            )
        sample_allocation = self.allocate_indices_to_children(
            samples_per_child, num_instances, num_children
        )

        return [
            self.create_sampling_result(data, labels, None, sample_indices)
            for sample_indices in sample_allocation
        ]


class CombinedSamplingStrategy(SamplingStrategy):
    """Combines feature and instance sampling.

    Applies both feature sampling and instance sampling to create
    child populations with reduced feature and instance sets.

    Args:
        feature_fraction: Multiplier for feature sample count. Default: 1.0.
        instance_fraction: Multiplier for instance sample count. Default: 1.0.
        replace: Whether to allow overlap between children when fractions <= 1.0.
            Default: False. Ignored for a dimension when its fraction > 1.0 (always True).

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> strategy = CombinedSamplingStrategy(
        ...     feature_fraction=0.5,
        ...     instance_fraction=0.5,
        ...     replace=False
        ... )
        >>> data = np.random.rand(100, 10) > 0.5
        >>> labels = np.random.randint(0, 2, 100)
        >>> results = strategy.sample(data, labels, num_features=10, num_children=3)
        >>> len(results)
        3
        >>> results[0].data.shape
        (50, 5)
    """

    def sample(
        self,
        data: ndarray,
        labels: ndarray,
        num_children: int,
    ) -> List[SamplingResult]:
        """Sample both features and instances for all children at once.

        Args:
            data: Training data as 2D boolean array (instances x features).
            labels: Training labels as 1D integer array.
            num_children: Number of child populations to create.

        Returns:
            List of SamplingResult, one per child, with both feature and instance
            subsets applied, containing both feature_indices and instance_indices.
        """
        num_instances, num_features = data.shape
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
            self.create_sampling_result(
                data,
                labels,
                feature_indices,
                sample_indices,
            )
            for sample_indices, feature_indices in zip(
                sample_allocation, feature_allocation
            )
        ]
