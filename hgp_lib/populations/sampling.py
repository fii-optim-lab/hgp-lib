"""Sampling strategies for hierarchical child population generation.

This module provides abstract base classes and data structures for sampling
data and features when creating child populations in hierarchical GP.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy import ndarray


@dataclass
class SamplingResult:
    """Result of a sampling operation containing sampled data and index mappings.

    Attributes:
        data: Sampled training data as 2D boolean ndarray (instances x features).
        labels: Sampled labels as 1D integer ndarray.
        feature_indices: Selected feature indices as 1D ndarray.
        instance_indices: Selected instance indices as 1D ndarray, or None for feature-only sampling.
    """

    data: ndarray
    labels: ndarray
    feature_indices: ndarray
    instance_indices: ndarray | None


class SamplingStrategy(ABC):
    """Abstract base class for data sampling strategies.

    Sampling strategies define how to select subsets of data and/or features
    for child populations in hierarchical GP.

    Attributes:
        MIN_FEATURES: Minimum number of features required in sampled result.
        MIN_INSTANCES: Minimum number of instances required in sampled result.
    """

    MIN_FEATURES = 2
    MIN_INSTANCES = 2

    @abstractmethod
    def sample(
        self,
        data: ndarray,
        labels: ndarray,
        num_features: int,
        num_children: int,
    ) -> SamplingResult:
        """Sample data and/or features from the training set.

        Args:
            data: Training data as 2D boolean array (instances x features).
            labels: Training labels as 1D integer array.
            num_features: Total number of features in the data.
            num_children: Number of child populations being created.

        Returns:
            SamplingResult containing sampled data, labels, and index mappings.
        """
        pass


class FeatureSamplingStrategy(SamplingStrategy):
    """Samples a subset of features from the training data.

    The number of features sampled is calculated as:
        base_count = ceil(num_features / num_children)
        sample_count = max(MIN_FEATURES, ceil(base_count * feature_fraction))

    Replacement behavior:
        - When feature_fraction > 1.0: always samples WITH replacement
        - When feature_fraction <= 1.0: uses the `replace` parameter (default: False)

    Args:
        feature_fraction: Multiplier for base sample count. Default: 1.0.
            - 1.0: Each child gets ~(num_features / num_children) features
            - >1.0: More features per child, with replacement (overlap)
            - <1.0: Fewer features per child
        replace: Whether to sample with replacement when feature_fraction <= 1.0.
            Default: False. Ignored when feature_fraction > 1.0 (always True).

    Examples:
        >>> import numpy as np
        >>> strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        >>> data = np.random.rand(100, 10) > 0.5
        >>> labels = np.random.randint(0, 2, 100)
        >>> result = strategy.sample(data, labels, num_features=10, num_children=3)
        >>> len(result.feature_indices)  # ~ceil(10/3) = 4 features
        4
    """

    def __init__(self, feature_fraction: float = 1.0, replace: bool = False):
        """Initialize FeatureSamplingStrategy.

        Args:
            feature_fraction: Multiplier for base sample count. Must be > 0.
            replace: Whether to sample with replacement when feature_fraction <= 1.0.

        Raises:
            ValueError: If feature_fraction is <= 0.
        """
        if feature_fraction <= 0:
            raise ValueError("feature_fraction must be > 0")
        self.feature_fraction = feature_fraction
        self.replace = replace

    def sample(
        self,
        data: ndarray,
        labels: ndarray,
        num_features: int,
        num_children: int,
    ) -> SamplingResult:
        """Sample a subset of features from the training data.

        Args:
            data: Training data as 2D boolean array (instances x features).
            labels: Training labels as 1D integer array.
            num_features: Total number of features in the data.
            num_children: Number of child populations being created.

        Returns:
            SamplingResult with sampled feature columns, all instances preserved,
            and instance_indices set to None.
        """
        base_count = ceil(num_features / num_children)
        sample_count = max(self.MIN_FEATURES, ceil(base_count * self.feature_fraction))

        # When fraction > 1.0, replacement is mandatory; otherwise use the parameter
        use_replace = True if self.feature_fraction > 1.0 else self.replace
        if not use_replace:
            sample_count = min(sample_count, num_features)

        feature_indices = np.random.choice(
            num_features,
            size=sample_count,
            replace=use_replace,
        )

        sampled_data = data[:, feature_indices]

        return SamplingResult(
            data=sampled_data,
            labels=labels,
            feature_indices=feature_indices,
            instance_indices=None,
        )


class InstanceSamplingStrategy(SamplingStrategy):
    """Samples a subset of instances from the training data.

    The number of instances sampled is calculated as:
        base_count = ceil(num_instances / num_children)
        sample_count = max(MIN_INSTANCES, ceil(base_count * instance_fraction))

    Replacement behavior:
        - When instance_fraction > 1.0: always samples WITH replacement
        - When instance_fraction <= 1.0: uses the `replace` parameter (default: False)

    Args:
        instance_fraction: Multiplier for base sample count. Default: 1.0.
            - 1.0: Each child gets ~(num_instances / num_children) instances
            - >1.0: More instances per child, with replacement (overlap)
            - <1.0: Fewer instances per child
        replace: Whether to sample with replacement when instance_fraction <= 1.0.
            Default: False. Ignored when instance_fraction > 1.0 (always True).

    Examples:
        >>> import numpy as np
        >>> strategy = InstanceSamplingStrategy(instance_fraction=1.0)
        >>> data = np.random.rand(100, 10) > 0.5
        >>> labels = np.random.randint(0, 2, 100)
        >>> result = strategy.sample(data, labels, num_features=10, num_children=3)
        >>> len(result.instance_indices)  # ~ceil(100/3) = 34 instances
        34
    """

    def __init__(self, instance_fraction: float = 1.0, replace: bool = False):
        """Initialize InstanceSamplingStrategy.

        Args:
            instance_fraction: Multiplier for base sample count. Must be > 0.
            replace: Whether to sample with replacement when instance_fraction <= 1.0.

        Raises:
            ValueError: If instance_fraction is <= 0.
        """
        if instance_fraction <= 0:
            raise ValueError("instance_fraction must be > 0")
        self.instance_fraction = instance_fraction
        self.replace = replace

    def sample(
        self,
        data: ndarray,
        labels: ndarray,
        num_features: int,
        num_children: int,
    ) -> SamplingResult:
        """Sample a subset of instances from the training data.

        Args:
            data: Training data as 2D boolean array (instances x features).
            labels: Training labels as 1D integer array.
            num_features: Total number of features in the data.
            num_children: Number of child populations being created.

        Returns:
            SamplingResult with sampled instance rows, all features preserved,
            and feature_indices containing all original indices.
        """
        num_instances = len(data)
        base_count = ceil(num_instances / num_children)
        sample_count = max(
            self.MIN_INSTANCES, ceil(base_count * self.instance_fraction)
        )

        # When fraction > 1.0, replacement is mandatory; otherwise use the parameter
        use_replace = True if self.instance_fraction > 1.0 else self.replace
        if not use_replace:
            sample_count = min(sample_count, num_instances)

        instance_indices = np.random.choice(
            num_instances,
            size=sample_count,
            replace=use_replace,
        )

        sampled_data = data[instance_indices, :]
        sampled_labels = labels[instance_indices]

        return SamplingResult(
            data=sampled_data,
            labels=sampled_labels,
            feature_indices=np.arange(num_features),
            instance_indices=instance_indices,
        )


class CombinedSamplingStrategy(SamplingStrategy):
    """Combines feature and instance sampling.

    Applies both feature sampling and instance sampling to create
    child populations with reduced feature and instance sets.

    The number of features sampled is calculated as:
        base_count = ceil(num_features / num_children)
        sample_count = max(MIN_FEATURES, ceil(base_count * feature_fraction))

    The number of instances sampled is calculated as:
        base_count = ceil(num_instances / num_children)
        sample_count = max(MIN_INSTANCES, ceil(base_count * instance_fraction))

    Replacement behavior for both dimensions:
        - When fraction > 1.0: always samples WITH replacement
        - When fraction <= 1.0: samples WITHOUT replacement

    Args:
        feature_fraction: Multiplier for feature sample count. Default: 1.0.
        instance_fraction: Multiplier for instance sample count. Default: 1.0.

    Examples:
        >>> import numpy as np
        >>> strategy = CombinedSamplingStrategy(
        ...     feature_fraction=1.0,
        ...     instance_fraction=1.0
        ... )
        >>> data = np.random.rand(100, 10) > 0.5
        >>> labels = np.random.randint(0, 2, 100)
        >>> result = strategy.sample(data, labels, num_features=10, num_children=3)
        >>> result.data.shape  # (~34 instances, ~4 features)
        (34, 4)
    """

    def __init__(
        self,
        feature_fraction: float = 1.0,
        instance_fraction: float = 1.0,
    ):
        """Initialize CombinedSamplingStrategy.

        Args:
            feature_fraction: Multiplier for feature sample count. Must be > 0.
            instance_fraction: Multiplier for instance sample count. Must be > 0.

        Raises:
            ValueError: If feature_fraction or instance_fraction is <= 0.
        """
        if feature_fraction <= 0:
            raise ValueError("feature_fraction must be > 0")
        if instance_fraction <= 0:
            raise ValueError("instance_fraction must be > 0")
        self.feature_fraction = feature_fraction
        self.instance_fraction = instance_fraction

    def sample(
        self,
        data: ndarray,
        labels: ndarray,
        num_features: int,
        num_children: int,
    ) -> SamplingResult:
        """Sample both features and instances from the training data.

        Args:
            data: Training data as 2D boolean array (instances x features).
            labels: Training labels as 1D integer array.
            num_features: Total number of features in the data.
            num_children: Number of child populations being created.

        Returns:
            SamplingResult with both feature and instance subsets applied,
            containing both feature_indices and instance_indices.
        """
        num_instances = len(data)

        # Feature sampling
        feature_base = ceil(num_features / num_children)
        feature_count = max(
            self.MIN_FEATURES, ceil(feature_base * self.feature_fraction)
        )
        feature_replace = self.feature_fraction > 1.0
        if not feature_replace:
            feature_count = min(feature_count, num_features)

        feature_indices = np.random.choice(
            num_features,
            size=feature_count,
            replace=feature_replace,
        )

        # Instance sampling
        instance_base = ceil(num_instances / num_children)
        instance_count = max(
            self.MIN_INSTANCES, ceil(instance_base * self.instance_fraction)
        )
        instance_replace = self.instance_fraction > 1.0
        if not instance_replace:
            instance_count = min(instance_count, num_instances)

        instance_indices = np.random.choice(
            num_instances,
            size=instance_count,
            replace=instance_replace,
        )

        # Use np.ix_ for 2D indexing
        sampled_data = data[np.ix_(instance_indices, feature_indices)]
        sampled_labels = labels[instance_indices]

        return SamplingResult(
            data=sampled_data,
            labels=sampled_labels,
            feature_indices=feature_indices,
            instance_indices=instance_indices,
        )
