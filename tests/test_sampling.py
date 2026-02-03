import unittest
import random
import doctest

import numpy as np

from hgp_lib.populations.sampling import (
    SamplingResult,
    SamplingStrategy,
    FeatureSamplingStrategy,
    InstanceSamplingStrategy,
    CombinedSamplingStrategy,
)
import hgp_lib.populations.sampling


class MockSamplingStrategy(SamplingStrategy):
    """Concrete implementation for testing SamplingStrategy interface."""

    def __init__(self, sample_instances: bool = False):
        self.sample_instances = sample_instances

    def sample(self, data, labels, num_features, num_children):
        num_instances = len(data)

        feature_count = max(self.MIN_FEATURES, num_features // max(1, num_children))
        feature_count = min(feature_count, num_features)
        feature_indices = np.random.choice(
            num_features, size=feature_count, replace=False
        )

        if self.sample_instances:
            instance_count = max(
                self.MIN_INSTANCES, num_instances // max(1, num_children)
            )
            instance_count = min(instance_count, num_instances)
            instance_indices = np.random.choice(
                num_instances, size=instance_count, replace=False
            )
            sampled_data = data[np.ix_(instance_indices, feature_indices)]
            sampled_labels = labels[instance_indices]
        else:
            instance_indices = None
            sampled_data = data[:, feature_indices]
            sampled_labels = labels

        return SamplingResult(
            data=sampled_data,
            labels=sampled_labels,
            feature_indices=feature_indices,
            instance_indices=instance_indices,
        )


class TestSamplingResult(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        np.random.seed(42)

        self.train_data = np.array(
            [
                [True, False, True, False, True],
                [False, True, False, True, False],
                [True, True, False, False, True],
                [False, False, True, True, False],
            ]
        )
        self.train_labels = np.array([1, 0, 1, 0])
        self.num_features = 5

    def test_sampling_result_data_is_ndarray(self):
        strategy = MockSamplingStrategy()
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=2
        )

        self.assertIsInstance(result.data, np.ndarray, "data should be a numpy array")

    def test_sampling_result_labels_is_ndarray(self):
        strategy = MockSamplingStrategy()
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=2
        )

        self.assertIsInstance(
            result.labels, np.ndarray, "labels should be a numpy array"
        )

    def test_sampling_result_feature_indices_is_ndarray(self):
        strategy = MockSamplingStrategy()
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=2
        )

        self.assertIsInstance(
            result.feature_indices,
            np.ndarray,
            "feature_indices should be a numpy array",
        )

    def test_sampling_result_instance_indices_is_ndarray_or_none(self):
        strategy_no_instances = MockSamplingStrategy(sample_instances=False)
        result = strategy_no_instances.sample(
            self.train_data, self.train_labels, self.num_features, num_children=2
        )
        self.assertIsNone(
            result.instance_indices,
            "instance_indices should be None for feature-only sampling",
        )

        strategy_with_instances = MockSamplingStrategy(sample_instances=True)
        result = strategy_with_instances.sample(
            self.train_data, self.train_labels, self.num_features, num_children=2
        )
        self.assertIsInstance(
            result.instance_indices,
            np.ndarray,
            "instance_indices should be a numpy array when sampling instances",
        )

    def test_sample_returns_sampling_result(self):
        strategy = MockSamplingStrategy()
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=2
        )

        self.assertIsInstance(
            result, SamplingResult, "sample() should return a SamplingResult instance"
        )


if __name__ == "__main__":
    unittest.main()


class TestFeatureSamplingStrategy(unittest.TestCase):
    """Tests for FeatureSamplingStrategy.

    Validates: Requirements 1.5, 3.1-3.9
    """

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

        self.train_data = np.array(
            [
                [True, False, True, False, True, False, True, False, True, False],
                [False, True, False, True, False, True, False, True, False, True],
                [True, True, False, False, True, True, False, False, True, True],
                [False, False, True, True, False, False, True, True, False, False],
            ]
        )
        self.train_labels = np.array([1, 0, 1, 0])
        self.num_features = 10
        self.num_instances = 4

    def test_constructor_validation(self):
        """Test constructor parameter validation."""
        with self.subTest("feature_fraction must be > 0"):
            with self.assertRaises(ValueError) as e:
                FeatureSamplingStrategy(feature_fraction=0)
            self.assertIn("feature_fraction must be > 0", str(e.exception))

        with self.subTest("negative feature_fraction raises ValueError"):
            with self.assertRaises(ValueError) as e:
                FeatureSamplingStrategy(feature_fraction=-0.5)
            self.assertIn("feature_fraction must be > 0", str(e.exception))

        with self.subTest("valid feature_fraction"):
            strategy = FeatureSamplingStrategy(feature_fraction=1.0)
            self.assertEqual(strategy.feature_fraction, 1.0)
            self.assertFalse(strategy.replace)

        with self.subTest("valid feature_fraction with replace=True"):
            strategy = FeatureSamplingStrategy(feature_fraction=0.5, replace=True)
            self.assertEqual(strategy.feature_fraction, 0.5)
            self.assertTrue(strategy.replace)

    def test_feature_sampling_preserves_instances(self):
        """Feature sampling preserves all instances.

        The returned SamplingResult SHALL have instance_indices set to None
        and the data SHALL have the same number of rows as the input.
        """
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=3
        )

        with self.subTest("instance_indices is None"):
            self.assertIsNone(result.instance_indices)

        with self.subTest("number of rows preserved"):
            self.assertEqual(result.data.shape[0], self.num_instances)

        with self.subTest("labels unchanged"):
            self.assertEqual(len(result.labels), self.num_instances)
            np.testing.assert_array_equal(result.labels, self.train_labels)

    def test_sample_count_calculation(self):
        """Sample count calculation follows formula.

        base_count = ceil(num_features / num_children)
        sample_count = max(MIN_FEATURES, ceil(base_count * fraction))
        """
        test_cases = [
            # (num_features, num_children, fraction, expected_count)
            (10, 3, 1.0, 4),  # ceil(10/3) = 4, ceil(4*1.0) = 4
            (10, 2, 1.0, 5),  # ceil(10/2) = 5, ceil(5*1.0) = 5
            (10, 5, 1.0, 2),  # ceil(10/5) = 2, max(2, ceil(2*1.0)) = 2
            (10, 3, 0.5, 2),  # ceil(10/3) = 4, max(2, ceil(4*0.5)) = 2
            (10, 3, 1.5, 6),  # ceil(10/3) = 4, ceil(4*1.5) = 6 (with replacement)
        ]

        for num_features, num_children, fraction, expected in test_cases:
            with self.subTest(
                num_features=num_features,
                num_children=num_children,
                fraction=fraction,
            ):
                np.random.seed(42)
                data = np.random.rand(10, num_features) > 0.5
                labels = np.random.randint(0, 2, 10)

                strategy = FeatureSamplingStrategy(feature_fraction=fraction)
                result = strategy.sample(data, labels, num_features, num_children)

                self.assertEqual(
                    len(result.feature_indices),
                    expected,
                    f"Expected {expected} features for fraction={fraction}",
                )
                self.assertEqual(result.data.shape[1], expected)

    def test_no_replacement_when_fraction_lte_1_and_replace_false(self):
        """No replacement when fraction <= 1.0 and replace=False.

        Sampled indices SHALL contain no duplicates.
        """
        strategy = FeatureSamplingStrategy(feature_fraction=0.8, replace=False)

        for seed in range(10):
            with self.subTest(seed=seed):
                np.random.seed(seed)
                result = strategy.sample(
                    self.train_data,
                    self.train_labels,
                    self.num_features,
                    num_children=3,
                )

                unique_indices = np.unique(result.feature_indices)
                self.assertEqual(
                    len(unique_indices),
                    len(result.feature_indices),
                    "No duplicates when replace=False",
                )

    def test_replacement_allowed_when_replace_true(self):
        """Replacement allowed when replace=True.

        When fraction <= 1.0 AND replace=True, sampling uses replacement.
        """
        # Use a small number of features to increase chance of duplicates
        small_data = self.train_data[:, :3]
        strategy = FeatureSamplingStrategy(feature_fraction=1.0, replace=True)

        # With replacement, we can sample more than num_features
        # The sample count is ceil(3/2) = 2, but with replace=True duplicates are allowed
        result = strategy.sample(small_data, self.train_labels, 3, num_children=2)

        # All indices should be valid (0, 1, or 2)
        self.assertTrue(all(0 <= idx < 3 for idx in result.feature_indices))

    def test_replacement_mandatory_when_fraction_gt_1(self):
        """Replacement mandatory when fraction > 1.0.

        Sampling uses replacement (duplicates MAY occur).
        """
        strategy = FeatureSamplingStrategy(feature_fraction=1.5)

        np.random.seed(42)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=3
        )

        # base_count = ceil(10/3) = 4, sample_count = ceil(4*1.5) = 6
        # With only 10 features and 6 samples, replacement is needed
        self.assertEqual(len(result.feature_indices), 6)

        # All indices should be valid
        self.assertTrue(
            all(0 <= idx < self.num_features for idx in result.feature_indices)
        )

    def test_minimum_count_enforcement(self):
        """Minimum feature count enforcement.

        The sampled result SHALL contain at least MIN_FEATURES features.
        """
        # Use many children and low fraction to try to get below minimum
        strategy = FeatureSamplingStrategy(feature_fraction=0.1)

        np.random.seed(42)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=20
        )

        self.assertGreaterEqual(
            len(result.feature_indices),
            SamplingStrategy.MIN_FEATURES,
            f"Should have at least {SamplingStrategy.MIN_FEATURES} features",
        )
        self.assertGreaterEqual(result.data.shape[1], SamplingStrategy.MIN_FEATURES)

    def test_invalid_fraction_rejection(self):
        """Invalid fraction rejection.

        Providing a fraction <= 0 SHALL raise a ValueError.
        """
        with self.subTest("fraction = 0"):
            with self.assertRaises(ValueError) as e:
                FeatureSamplingStrategy(feature_fraction=0)
            self.assertIn("feature_fraction must be > 0", str(e.exception))

        with self.subTest("fraction = -0.5"):
            with self.assertRaises(ValueError) as e:
                FeatureSamplingStrategy(feature_fraction=-0.5)
            self.assertIn("feature_fraction must be > 0", str(e.exception))

        with self.subTest("fraction = -1"):
            with self.assertRaises(ValueError) as e:
                FeatureSamplingStrategy(feature_fraction=-1)
            self.assertIn("feature_fraction must be > 0", str(e.exception))

    def test_data_shape_consistency(self):
        """Data shape consistency.

        result.data.shape[1] == len(result.feature_indices)
        result.data.shape[0] == len(result.labels)
        """
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=3
        )

        with self.subTest("columns match feature_indices"):
            self.assertEqual(result.data.shape[1], len(result.feature_indices))

        with self.subTest("rows match labels"):
            self.assertEqual(result.data.shape[0], len(result.labels))

    def test_sampled_data_matches_indices(self):
        """Verify that sampled data columns match the feature_indices."""
        np.random.seed(42)
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=3
        )

        # Verify each column in sampled data matches the original column
        for i, feature_idx in enumerate(result.feature_indices):
            np.testing.assert_array_equal(
                result.data[:, i],
                self.train_data[:, feature_idx],
                f"Column {i} should match original feature {feature_idx}",
            )


class TestInstanceSamplingStrategy(unittest.TestCase):
    """Tests for InstanceSamplingStrategy.

    Validates: Requirements 1.6, 4.1-4.9
    """

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

        self.train_data = np.array(
            [
                [True, False, True, False, True],
                [False, True, False, True, False],
                [True, True, False, False, True],
                [False, False, True, True, False],
                [True, False, False, True, True],
                [False, True, True, False, False],
                [True, True, True, True, True],
                [False, False, False, False, False],
                [True, False, True, True, False],
                [False, True, False, False, True],
            ]
        )
        self.train_labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.num_features = 5
        self.num_instances = 10

    def test_constructor_validation(self):
        """Test constructor parameter validation."""
        with self.subTest("instance_fraction must be > 0"):
            with self.assertRaises(ValueError) as e:
                InstanceSamplingStrategy(instance_fraction=0)
            self.assertIn("instance_fraction must be > 0", str(e.exception))

        with self.subTest("negative instance_fraction raises ValueError"):
            with self.assertRaises(ValueError) as e:
                InstanceSamplingStrategy(instance_fraction=-0.5)
            self.assertIn("instance_fraction must be > 0", str(e.exception))

        with self.subTest("valid instance_fraction"):
            strategy = InstanceSamplingStrategy(instance_fraction=1.0)
            self.assertEqual(strategy.instance_fraction, 1.0)
            self.assertFalse(strategy.replace)

        with self.subTest("valid instance_fraction with replace=True"):
            strategy = InstanceSamplingStrategy(instance_fraction=0.5, replace=True)
            self.assertEqual(strategy.instance_fraction, 0.5)
            self.assertTrue(strategy.replace)

    def test_instance_sampling_preserves_features(self):
        """Instance sampling preserves all features.

        The returned SamplingResult SHALL have feature_indices containing
        all original feature indices (0 to num_features-1).
        """
        strategy = InstanceSamplingStrategy(instance_fraction=1.0)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=3
        )

        with self.subTest("feature_indices contains all features"):
            expected_indices = np.arange(self.num_features)
            np.testing.assert_array_equal(result.feature_indices, expected_indices)

        with self.subTest("number of columns preserved"):
            self.assertEqual(result.data.shape[1], self.num_features)

    def test_sample_count_calculation(self):
        """Sample count calculation follows formula.

        base_count = ceil(num_instances / num_children)
        sample_count = max(MIN_INSTANCES, ceil(base_count * fraction))
        """
        test_cases = [
            # (num_instances, num_children, fraction, expected_count)
            (10, 3, 1.0, 4),  # ceil(10/3) = 4, ceil(4*1.0) = 4
            (10, 2, 1.0, 5),  # ceil(10/2) = 5, ceil(5*1.0) = 5
            (10, 5, 1.0, 2),  # ceil(10/5) = 2, max(2, ceil(2*1.0)) = 2
            (10, 3, 0.5, 2),  # ceil(10/3) = 4, max(2, ceil(4*0.5)) = 2
            (10, 3, 1.5, 6),  # ceil(10/3) = 4, ceil(4*1.5) = 6 (with replacement)
        ]

        for num_instances, num_children, fraction, expected in test_cases:
            with self.subTest(
                num_instances=num_instances,
                num_children=num_children,
                fraction=fraction,
            ):
                np.random.seed(42)
                data = np.random.rand(num_instances, 5) > 0.5
                labels = np.random.randint(0, 2, num_instances)

                strategy = InstanceSamplingStrategy(instance_fraction=fraction)
                result = strategy.sample(data, labels, 5, num_children)

                self.assertEqual(
                    len(result.instance_indices),
                    expected,
                    f"Expected {expected} instances for fraction={fraction}",
                )
                self.assertEqual(result.data.shape[0], expected)

    def test_no_replacement_when_fraction_lte_1_and_replace_false(self):
        """No replacement when fraction <= 1.0 and replace=False.

        Sampled indices SHALL contain no duplicates.
        """
        strategy = InstanceSamplingStrategy(instance_fraction=0.8, replace=False)

        for seed in range(10):
            with self.subTest(seed=seed):
                np.random.seed(seed)
                result = strategy.sample(
                    self.train_data,
                    self.train_labels,
                    self.num_features,
                    num_children=3,
                )

                unique_indices = np.unique(result.instance_indices)
                self.assertEqual(
                    len(unique_indices),
                    len(result.instance_indices),
                    "No duplicates when replace=False",
                )

    def test_replacement_allowed_when_replace_true(self):
        """Replacement allowed when replace=True.

        When fraction <= 1.0 AND replace=True, sampling uses replacement.
        """
        # Use a small number of instances to increase chance of duplicates
        small_data = self.train_data[:3, :]
        small_labels = self.train_labels[:3]
        strategy = InstanceSamplingStrategy(instance_fraction=1.0, replace=True)

        result = strategy.sample(
            small_data, small_labels, self.num_features, num_children=2
        )

        # All indices should be valid (0, 1, or 2)
        self.assertTrue(all(0 <= idx < 3 for idx in result.instance_indices))

    def test_replacement_mandatory_when_fraction_gt_1(self):
        """Replacement mandatory when fraction > 1.0.

        Sampling uses replacement (duplicates MAY occur).
        """
        strategy = InstanceSamplingStrategy(instance_fraction=1.5)

        np.random.seed(42)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=3
        )

        # base_count = ceil(10/3) = 4, sample_count = ceil(4*1.5) = 6
        self.assertEqual(len(result.instance_indices), 6)

        # All indices should be valid
        self.assertTrue(
            all(0 <= idx < self.num_instances for idx in result.instance_indices)
        )

    def test_minimum_count_enforcement(self):
        """Minimum instance count enforcement.

        The sampled result SHALL contain at least MIN_INSTANCES instances.
        """
        # Use many children and low fraction to try to get below minimum
        strategy = InstanceSamplingStrategy(instance_fraction=0.1)

        np.random.seed(42)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=20
        )

        self.assertGreaterEqual(
            len(result.instance_indices),
            SamplingStrategy.MIN_INSTANCES,
            f"Should have at least {SamplingStrategy.MIN_INSTANCES} instances",
        )
        self.assertGreaterEqual(result.data.shape[0], SamplingStrategy.MIN_INSTANCES)

    def test_invalid_fraction_rejection(self):
        """Invalid fraction rejection.

        Providing a fraction <= 0 SHALL raise a ValueError.
        """
        with self.subTest("fraction = 0"):
            with self.assertRaises(ValueError) as e:
                InstanceSamplingStrategy(instance_fraction=0)
            self.assertIn("instance_fraction must be > 0", str(e.exception))

        with self.subTest("fraction = -0.5"):
            with self.assertRaises(ValueError) as e:
                InstanceSamplingStrategy(instance_fraction=-0.5)
            self.assertIn("instance_fraction must be > 0", str(e.exception))

        with self.subTest("fraction = -1"):
            with self.assertRaises(ValueError) as e:
                InstanceSamplingStrategy(instance_fraction=-1)
            self.assertIn("instance_fraction must be > 0", str(e.exception))

    def test_data_shape_consistency(self):
        """Data shape consistency.

        result.data.shape[0] == len(result.instance_indices)
        result.data.shape[0] == len(result.labels)
        result.data.shape[1] == len(result.feature_indices)
        """
        strategy = InstanceSamplingStrategy(instance_fraction=1.0)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=3
        )

        with self.subTest("rows match instance_indices"):
            self.assertEqual(result.data.shape[0], len(result.instance_indices))

        with self.subTest("rows match labels"):
            self.assertEqual(result.data.shape[0], len(result.labels))

        with self.subTest("columns match feature_indices"):
            self.assertEqual(result.data.shape[1], len(result.feature_indices))

    def test_sampled_data_matches_indices(self):
        """Verify that sampled data rows match the instance_indices."""
        np.random.seed(42)
        strategy = InstanceSamplingStrategy(instance_fraction=1.0)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=3
        )

        # Verify each row in sampled data matches the original row
        for i, instance_idx in enumerate(result.instance_indices):
            np.testing.assert_array_equal(
                result.data[i, :],
                self.train_data[instance_idx, :],
                f"Row {i} should match original instance {instance_idx}",
            )
            self.assertEqual(
                result.labels[i],
                self.train_labels[instance_idx],
                f"Label {i} should match original label at {instance_idx}",
            )


class TestCombinedSamplingStrategy(unittest.TestCase):
    """Tests for CombinedSamplingStrategy."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

        self.train_data = np.array(
            [
                [True, False, True, False, True, False, True, False, True, False],
                [False, True, False, True, False, True, False, True, False, True],
                [True, True, False, False, True, True, False, False, True, True],
                [False, False, True, True, False, False, True, True, False, False],
                [True, False, False, True, True, False, False, True, True, False],
                [False, True, True, False, False, True, True, False, False, True],
                [True, True, True, True, True, True, True, True, True, True],
                [False, False, False, False, False, False, False, False, False, False],
                [True, False, True, True, False, True, False, True, True, False],
                [False, True, False, False, True, False, True, False, False, True],
            ]
        )
        self.train_labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.num_features = 10
        self.num_instances = 10

    def test_constructor_validation(self):
        """Test constructor parameter validation."""
        with self.subTest("feature_fraction must be > 0"):
            with self.assertRaises(ValueError) as e:
                CombinedSamplingStrategy(feature_fraction=0)
            self.assertIn("feature_fraction must be > 0", str(e.exception))

        with self.subTest("instance_fraction must be > 0"):
            with self.assertRaises(ValueError) as e:
                CombinedSamplingStrategy(instance_fraction=0)
            self.assertIn("instance_fraction must be > 0", str(e.exception))

        with self.subTest("negative feature_fraction raises ValueError"):
            with self.assertRaises(ValueError) as e:
                CombinedSamplingStrategy(feature_fraction=-0.5)
            self.assertIn("feature_fraction must be > 0", str(e.exception))

        with self.subTest("negative instance_fraction raises ValueError"):
            with self.assertRaises(ValueError) as e:
                CombinedSamplingStrategy(instance_fraction=-0.5)
            self.assertIn("instance_fraction must be > 0", str(e.exception))

        with self.subTest("valid fractions"):
            strategy = CombinedSamplingStrategy(
                feature_fraction=1.0, instance_fraction=1.0
            )
            self.assertEqual(strategy.feature_fraction, 1.0)
            self.assertEqual(strategy.instance_fraction, 1.0)

    def test_combined_strategy_produces_both_indices(self):
        """Combined strategy produces both indices.

        The result SHALL have both feature_indices and instance_indices
        as non-None ndarrays.
        """
        strategy = CombinedSamplingStrategy(feature_fraction=1.0, instance_fraction=1.0)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=3
        )

        with self.subTest("feature_indices is not None"):
            self.assertIsNotNone(result.feature_indices)
            self.assertIsInstance(result.feature_indices, np.ndarray)

        with self.subTest("instance_indices is not None"):
            self.assertIsNotNone(result.instance_indices)
            self.assertIsInstance(result.instance_indices, np.ndarray)

    def test_data_shape_consistency(self):
        """Data shape consistency.

        - result.data.shape[1] == len(result.feature_indices)
        - result.data.shape[0] == len(result.labels)
        - result.data.shape[0] == len(result.instance_indices)
        """
        strategy = CombinedSamplingStrategy(feature_fraction=1.0, instance_fraction=1.0)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=3
        )

        with self.subTest("columns match feature_indices"):
            self.assertEqual(result.data.shape[1], len(result.feature_indices))

        with self.subTest("rows match labels"):
            self.assertEqual(result.data.shape[0], len(result.labels))

        with self.subTest("rows match instance_indices"):
            self.assertEqual(result.data.shape[0], len(result.instance_indices))

    def test_sampled_data_matches_indices(self):
        """Verify that sampled data matches both feature and instance indices."""
        np.random.seed(42)
        strategy = CombinedSamplingStrategy(feature_fraction=1.0, instance_fraction=1.0)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=3
        )

        for i, instance_idx in enumerate(result.instance_indices):
            for j, feature_idx in enumerate(result.feature_indices):
                self.assertEqual(
                    result.data[i, j],
                    self.train_data[instance_idx, feature_idx],
                    f"Cell ({i},{j}) should match original ({instance_idx},{feature_idx})",
                )
            self.assertEqual(
                result.labels[i],
                self.train_labels[instance_idx],
                f"Label {i} should match original label at {instance_idx}",
            )

    def test_sample_count_calculation(self):
        """Sample count calculation follows formula for both dimensions."""
        test_cases = [
            (10, 10, 3, 1.0, 1.0, 4, 4),
            (10, 10, 2, 1.0, 1.0, 5, 5),
            (10, 10, 5, 1.0, 1.0, 2, 2),
            (10, 10, 3, 0.5, 0.5, 2, 2),
            (10, 10, 3, 1.5, 1.5, 6, 6),
        ]

        for (
            num_feat,
            num_inst,
            num_children,
            feat_frac,
            inst_frac,
            exp_feat,
            exp_inst,
        ) in test_cases:
            with self.subTest(
                num_features=num_feat,
                num_instances=num_inst,
                num_children=num_children,
                feature_fraction=feat_frac,
                instance_fraction=inst_frac,
            ):
                np.random.seed(42)
                data = np.random.rand(num_inst, num_feat) > 0.5
                labels = np.random.randint(0, 2, num_inst)

                strategy = CombinedSamplingStrategy(
                    feature_fraction=feat_frac,
                    instance_fraction=inst_frac,
                )
                result = strategy.sample(data, labels, num_feat, num_children)

                self.assertEqual(len(result.feature_indices), exp_feat)
                self.assertEqual(len(result.instance_indices), exp_inst)
                self.assertEqual(result.data.shape, (exp_inst, exp_feat))

    def test_no_replacement_when_fraction_lte_1(self):
        """No replacement when fraction <= 1.0."""
        strategy = CombinedSamplingStrategy(feature_fraction=0.8, instance_fraction=0.8)

        for seed in range(10):
            with self.subTest(seed=seed):
                np.random.seed(seed)
                result = strategy.sample(
                    self.train_data,
                    self.train_labels,
                    self.num_features,
                    num_children=3,
                )

                unique_features = np.unique(result.feature_indices)
                self.assertEqual(len(unique_features), len(result.feature_indices))

                unique_instances = np.unique(result.instance_indices)
                self.assertEqual(len(unique_instances), len(result.instance_indices))

    def test_replacement_mandatory_when_fraction_gt_1(self):
        """Replacement mandatory when fraction > 1.0."""
        strategy = CombinedSamplingStrategy(feature_fraction=1.5, instance_fraction=1.5)

        np.random.seed(42)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=3
        )

        self.assertEqual(len(result.feature_indices), 6)
        self.assertEqual(len(result.instance_indices), 6)

        self.assertTrue(
            all(0 <= idx < self.num_features for idx in result.feature_indices)
        )
        self.assertTrue(
            all(0 <= idx < self.num_instances for idx in result.instance_indices)
        )

    def test_minimum_count_enforcement(self):
        """Minimum count enforcement for both dimensions."""
        strategy = CombinedSamplingStrategy(feature_fraction=0.1, instance_fraction=0.1)

        np.random.seed(42)
        result = strategy.sample(
            self.train_data, self.train_labels, self.num_features, num_children=20
        )

        self.assertGreaterEqual(
            len(result.feature_indices), SamplingStrategy.MIN_FEATURES
        )
        self.assertGreaterEqual(
            len(result.instance_indices), SamplingStrategy.MIN_INSTANCES
        )
        self.assertGreaterEqual(result.data.shape[0], SamplingStrategy.MIN_INSTANCES)
        self.assertGreaterEqual(result.data.shape[1], SamplingStrategy.MIN_FEATURES)

    def test_invalid_fraction_rejection(self):
        """Invalid fraction rejection."""
        with self.subTest("feature_fraction = 0"):
            with self.assertRaises(ValueError) as e:
                CombinedSamplingStrategy(feature_fraction=0)
            self.assertIn("feature_fraction must be > 0", str(e.exception))

        with self.subTest("instance_fraction = 0"):
            with self.assertRaises(ValueError) as e:
                CombinedSamplingStrategy(instance_fraction=0)
            self.assertIn("instance_fraction must be > 0", str(e.exception))

        with self.subTest("feature_fraction = -0.5"):
            with self.assertRaises(ValueError) as e:
                CombinedSamplingStrategy(feature_fraction=-0.5)
            self.assertIn("feature_fraction must be > 0", str(e.exception))

        with self.subTest("instance_fraction = -1"):
            with self.assertRaises(ValueError) as e:
                CombinedSamplingStrategy(instance_fraction=-1)
            self.assertIn("instance_fraction must be > 0", str(e.exception))


class TestSamplingDoctests(unittest.TestCase):
    """Test that all doctests in the sampling module pass."""

    def test_doctests(self):
        """Run all doctests in the sampling module."""
        result = doctest.testmod(hgp_lib.populations.sampling, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
