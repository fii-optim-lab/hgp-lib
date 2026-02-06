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
    ) -> List[SamplingResult]:
        """Sample data and/or features for child populations.

        The `replace` parameter controls overlap between children:
            - replace=False: No overlap (partitioning)
            - replace=True: Overlap allowed

        Args:
            data: Training data as 2D boolean array (instances x features).
            labels: Training labels as 1D integer array.
            num_features: Total number of features in the data.
            num_children: Number of child populations to create.

        Returns:
            List of SamplingResult, one per child (exactly `num_children` elements).
        """
        pass


class FeatureSamplingStrategy(SamplingStrategy):
    """Samples a subset of features from the training data.

    The number of features sampled per child is calculated as:
        base_count = ceil(num_features / num_children)
        sample_count = max(MIN_FEATURES, ceil(base_count * feature_fraction))

    Overlap behavior (controlled by `replace` parameter):
        - replace=False: No overlap between children (partitioning) - each feature
          appears in at most one child population
        - replace=True: Overlap allowed - same feature can appear in multiple children
        - When feature_fraction > 1.0: overlap is mandatory (equivalent to replace=True)

    Within each child, features are always unique (no duplicates within a single child).

    Args:
        feature_fraction: Multiplier for base sample count. Default: 1.0.
            - 1.0: Each child gets ~(num_features / num_children) features
            - >1.0: More features per child, with overlap between children
            - <1.0: Fewer features per child
        replace: Whether to allow overlap between children when feature_fraction <= 1.0.
            Default: False. Ignored when feature_fraction > 1.0 (always True).

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        >>> data = np.random.rand(100, 10) > 0.5
        >>> labels = np.random.randint(0, 2, 100)
        >>> results = strategy.sample(data, labels, num_features=10, num_children=3)
        >>> len(results)  # One result per child
        3
        >>> len(results[0].feature_indices)  # ~ceil(10/3) = 4 features
        4
    """

    def __init__(self, feature_fraction: float = 1.0, replace: bool = False):
        """Initialize FeatureSamplingStrategy.

        Args:
            feature_fraction: Multiplier for base sample count. Must be > 0.
            replace: Whether to allow overlap between children when feature_fraction <= 1.0.

        Raises:
            ValueError: If feature_fraction is <= 0.
        """
        if feature_fraction <= 0:
            raise ValueError("feature_fraction must be > 0")
        self.feature_fraction = feature_fraction
        self.replace = replace

    def _allocate_features(
        self,
        num_features: int,
        num_children: int,
        sample_count: int,
        allow_overlap: bool,
    ) -> List[ndarray]:
        """Allocate features for all children.

        Args:
            num_features: Total number of features available.
            num_children: Number of children to allocate for.
            sample_count: How many features per child.
            allow_overlap: If True, same feature can appear in multiple children.
                          If False, partitioning - each feature in at most one child.

        Returns:
            List of arrays with feature indices for each child.
        """
        if allow_overlap:
            # With overlap: each child gets sample_count unique features
            # but the same feature can appear in multiple children
            return [
                np.random.choice(num_features, size=sample_count, replace=False)
                for _ in range(num_children)
            ]
        else:
            # Without overlap: partitioning - each feature in at most one child
            all_features = np.random.permutation(num_features)
            allocations = []
            for i in range(num_children):
                start = i * sample_count
                end = start + sample_count
                if end <= num_features:
                    allocations.append(all_features[start:end])
                else:
                    # Wrap around to ensure sample_count features
                    indices = np.concatenate(
                        [all_features[start:], all_features[: end - num_features]]
                    )
                    allocations.append(indices)
            return allocations

    def sample(
        self,
        data: ndarray,
        labels: ndarray,
        num_features: int,
        num_children: int,
    ) -> List[SamplingResult]:
        """Sample features for child populations.

        Args:
            data: Training data as 2D boolean array (instances x features).
            labels: Training labels as 1D integer array.
            num_features: Total number of features in the data.
            num_children: Number of child populations to create.

        Returns:
            List of SamplingResult, one per child, with sampled feature columns,
            all instances preserved, and instance_indices set to None.
        """
        base_count = ceil(num_features / num_children)
        sample_count = max(self.MIN_FEATURES, ceil(base_count * self.feature_fraction))

        # When fraction > 1.0, overlap is mandatory; otherwise use the replace parameter
        allow_overlap = self.replace or self.feature_fraction > 1.0

        # Ensure sample_count doesn't exceed num_features (can't sample more unique items than exist)
        sample_count = min(sample_count, num_features)

        # Allocate features for all children
        feature_allocations = self._allocate_features(
            num_features, num_children, sample_count, allow_overlap
        )

        results = []
        for feature_indices in feature_allocations:
            sampled_data = data[:, feature_indices]
            results.append(
                SamplingResult(
                    data=sampled_data,
                    labels=labels,
                    feature_indices=feature_indices,
                    instance_indices=None,
                    feature_mapping={
                        i: int(idx) for i, idx in enumerate(feature_indices)
                    },
                )
            )
        return results


class InstanceSamplingStrategy(SamplingStrategy):
    """Samples a subset of instances from the training data.

    The number of instances sampled per child is calculated as:
        base_count = ceil(num_instances / num_children)
        sample_count = max(MIN_INSTANCES, ceil(base_count * instance_fraction))

    Overlap behavior (controlled by `replace` parameter):
        - replace=False: No overlap between children (partitioning) - each instance
          appears in at most one child population
        - replace=True: Overlap allowed - same instance can appear in multiple children
        - When instance_fraction > 1.0: overlap is mandatory (equivalent to replace=True)

    Within each child, instances are always unique (no duplicates within a single child).

    Args:
        instance_fraction: Multiplier for base sample count. Default: 1.0.
            - 1.0: Each child gets ~(num_instances / num_children) instances
            - >1.0: More instances per child, with overlap between children
            - <1.0: Fewer instances per child
        replace: Whether to allow overlap between children when instance_fraction <= 1.0.
            Default: False. Ignored when instance_fraction > 1.0 (always True).

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> strategy = InstanceSamplingStrategy(instance_fraction=1.0)
        >>> data = np.random.rand(100, 10) > 0.5
        >>> labels = np.random.randint(0, 2, 100)
        >>> results = strategy.sample(data, labels, num_features=10, num_children=3)
        >>> len(results)  # One result per child
        3
        >>> len(results[0].instance_indices)  # ~ceil(100/3) = 34 instances
        34
    """

    def __init__(self, instance_fraction: float = 1.0, replace: bool = False):
        """Initialize InstanceSamplingStrategy.

        Args:
            instance_fraction: Multiplier for base sample count. Must be > 0.
            replace: Whether to allow overlap between children when instance_fraction <= 1.0.

        Raises:
            ValueError: If instance_fraction is <= 0.
        """
        if instance_fraction <= 0:
            raise ValueError("instance_fraction must be > 0")
        self.instance_fraction = instance_fraction
        self.replace = replace

    def _allocate_instances(
        self,
        num_instances: int,
        num_children: int,
        sample_count: int,
        allow_overlap: bool,
    ) -> List[ndarray]:
        """Allocate instances for all children.

        Args:
            num_instances: Total number of instances available.
            num_children: Number of children to allocate for.
            sample_count: How many instances per child.
            allow_overlap: If True, same instance can appear in multiple children.
                          If False, partitioning - each instance in at most one child.

        Returns:
            List of arrays with instance indices for each child.
        """
        if allow_overlap:
            # With overlap: each child gets sample_count unique instances
            # but the same instance can appear in multiple children
            return [
                np.random.choice(num_instances, size=sample_count, replace=False)
                for _ in range(num_children)
            ]
        else:
            # Without overlap: partitioning - each instance in at most one child
            all_instances = np.random.permutation(num_instances)
            allocations = []
            for i in range(num_children):
                start = i * sample_count
                end = start + sample_count
                if end <= num_instances:
                    allocations.append(all_instances[start:end])
                else:
                    # Wrap around to ensure sample_count instances
                    indices = np.concatenate(
                        [all_instances[start:], all_instances[: end - num_instances]]
                    )
                    allocations.append(indices)
            return allocations

    def sample(
        self,
        data: ndarray,
        labels: ndarray,
        num_features: int,
        num_children: int,
    ) -> List[SamplingResult]:
        """Sample instances for child populations.

        Args:
            data: Training data as 2D boolean array (instances x features).
            labels: Training labels as 1D integer array.
            num_features: Total number of features in the data.
            num_children: Number of child populations to create.

        Returns:
            List of SamplingResult, one per child, with sampled instance rows,
            all features preserved, and feature_mapping set to None.
        """
        num_instances = len(data)
        base_count = ceil(num_instances / num_children)
        sample_count = max(
            self.MIN_INSTANCES, ceil(base_count * self.instance_fraction)
        )

        # When fraction > 1.0, overlap is mandatory; otherwise use the replace parameter
        allow_overlap = self.replace or self.instance_fraction > 1.0

        # Ensure sample_count doesn't exceed num_instances (can't sample more unique items than exist)
        sample_count = min(sample_count, num_instances)

        # Allocate instances for all children
        instance_allocations = self._allocate_instances(
            num_instances, num_children, sample_count, allow_overlap
        )

        results = []
        for instance_indices in instance_allocations:
            sampled_data = data[instance_indices, :]
            sampled_labels = labels[instance_indices]
            results.append(
                SamplingResult(
                    data=sampled_data,
                    labels=sampled_labels,
                    feature_indices=np.arange(num_features),
                    instance_indices=instance_indices,
                    feature_mapping=None,
                )
            )
        return results


class CombinedSamplingStrategy(SamplingStrategy):
    """Combines feature and instance sampling.

    Applies both feature sampling and instance sampling to create
    child populations with reduced feature and instance sets.

    The number of features sampled per child is calculated as:
        base_count = ceil(num_features / num_children)
        sample_count = max(MIN_FEATURES, ceil(base_count * feature_fraction))

    The number of instances sampled per child is calculated as:
        base_count = ceil(num_instances / num_children)
        sample_count = max(MIN_INSTANCES, ceil(base_count * instance_fraction))

    Overlap behavior (controlled by `replace` parameter):
        - replace=False: No overlap between children (partitioning) - each feature/instance
          appears in at most one child population
        - replace=True: Overlap allowed - same feature/instance can appear in multiple children
        - When fraction > 1.0: overlap is mandatory for that dimension (equivalent to replace=True)

    Within each child, features and instances are always unique (no duplicates within a single child).

    Args:
        feature_fraction: Multiplier for feature sample count. Default: 1.0.
        instance_fraction: Multiplier for instance sample count. Default: 1.0.
        replace: Whether to allow overlap between children when fractions <= 1.0.
            Default: False. Ignored for a dimension when its fraction > 1.0 (always True).

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> strategy = CombinedSamplingStrategy(
        ...     feature_fraction=1.0,
        ...     instance_fraction=1.0,
        ...     replace=False
        ... )
        >>> data = np.random.rand(100, 10) > 0.5
        >>> labels = np.random.randint(0, 2, 100)
        >>> results = strategy.sample(data, labels, num_features=10, num_children=3)
        >>> len(results)  # One result per child
        3
        >>> results[0].data.shape  # (~34 instances, ~4 features)
        (34, 4)
    """

    def __init__(
        self,
        feature_fraction: float = 1.0,
        instance_fraction: float = 1.0,
        replace: bool = False,
    ):
        """Initialize CombinedSamplingStrategy.

        Args:
            feature_fraction: Multiplier for feature sample count. Must be > 0.
            instance_fraction: Multiplier for instance sample count. Must be > 0.
            replace: Whether to allow overlap between children when fractions <= 1.0.

        Raises:
            ValueError: If feature_fraction or instance_fraction is <= 0.
        """
        if feature_fraction <= 0:
            raise ValueError("feature_fraction must be > 0")
        if instance_fraction <= 0:
            raise ValueError("instance_fraction must be > 0")
        self.feature_fraction = feature_fraction
        self.instance_fraction = instance_fraction
        self.replace = replace

    def _allocate(
        self,
        total: int,
        num_children: int,
        sample_count: int,
        allow_overlap: bool,
    ) -> List[ndarray]:
        """Generic allocation method for features or instances.

        Args:
            total: Total number of items available (features or instances).
            num_children: Number of children to allocate for.
            sample_count: How many items per child.
            allow_overlap: If True, same item can appear in multiple children.
                          If False, partitioning - each item in at most one child.

        Returns:
            List of arrays with indices for each child.
        """
        if allow_overlap:
            # With overlap: each child gets sample_count unique items
            # but the same item can appear in multiple children
            return [
                np.random.choice(total, size=sample_count, replace=False)
                for _ in range(num_children)
            ]
        else:
            # Without overlap: partitioning - each item in at most one child
            all_items = np.random.permutation(total)
            allocations = []
            for i in range(num_children):
                start = i * sample_count
                end = start + sample_count
                if end <= total:
                    allocations.append(all_items[start:end])
                else:
                    # Wrap around to ensure sample_count items
                    indices = np.concatenate(
                        [all_items[start:], all_items[: end - total]]
                    )
                    allocations.append(indices)
            return allocations

    def sample(
        self,
        data: ndarray,
        labels: ndarray,
        num_features: int,
        num_children: int,
    ) -> List[SamplingResult]:
        """Sample both features and instances for all children at once.

        Args:
            data: Training data as 2D boolean array (instances x features).
            labels: Training labels as 1D integer array.
            num_features: Total number of features in the data.
            num_children: Number of child populations to create.

        Returns:
            List of SamplingResult, one per child, with both feature and instance
            subsets applied, containing both feature_indices and instance_indices.
        """
        num_instances = len(data)

        # Calculate feature sample count
        feature_base = ceil(num_features / num_children)
        feature_count = max(
            self.MIN_FEATURES, ceil(feature_base * self.feature_fraction)
        )

        # Calculate instance sample count
        instance_base = ceil(num_instances / num_children)
        instance_count = max(
            self.MIN_INSTANCES, ceil(instance_base * self.instance_fraction)
        )

        # Determine overlap behavior for each dimension
        feature_overlap = self.replace or self.feature_fraction > 1.0
        instance_overlap = self.replace or self.instance_fraction > 1.0

        # Ensure sample_count doesn't exceed total (can't sample more unique items than exist)
        feature_count = min(feature_count, num_features)
        instance_count = min(instance_count, num_instances)

        # Allocate features and instances for all children
        feature_allocations = self._allocate(
            num_features, num_children, feature_count, feature_overlap
        )
        instance_allocations = self._allocate(
            num_instances, num_children, instance_count, instance_overlap
        )

        # Build results for each child
        results = []
        for feat_idx, inst_idx in zip(feature_allocations, instance_allocations):
            sampled_data = data[np.ix_(inst_idx, feat_idx)]
            sampled_labels = labels[inst_idx]
            results.append(
                SamplingResult(
                    data=sampled_data,
                    labels=sampled_labels,
                    feature_indices=feat_idx,
                    instance_indices=inst_idx,
                    feature_mapping={i: int(idx) for i, idx in enumerate(feat_idx)},
                )
            )
        return results
