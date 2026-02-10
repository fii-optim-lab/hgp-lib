"""Tests for sampling strategies."""

import unittest
import numpy as np

from hgp_lib.populations.sampling import (
    FeatureSamplingStrategy,
    InstanceSamplingStrategy,
    CombinedSamplingStrategy,
)


class TestFeatureSamplingStrategy(unittest.TestCase):
    """Tests for FeatureSamplingStrategy."""

    def setUp(self):
        np.random.seed(42)
        self.data = np.random.rand(100, 20) > 0.5
        self.labels = np.random.randint(0, 2, 100)

    def test_returns_correct_number_of_results(self):
        """sample() returns exactly num_children results."""
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_children=5)
        self.assertEqual(len(results), 5)

    def test_each_result_has_unique_features(self):
        """Each child has unique feature indices (no duplicates within a child)."""
        strategy = FeatureSamplingStrategy(feature_fraction=1.0, replace=True)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            unique_count = len(np.unique(result.feature_indices))
            self.assertEqual(unique_count, len(result.feature_indices))

    def test_no_overlap_when_replace_false(self):
        """When replace=False, features don't overlap between children."""
        strategy = FeatureSamplingStrategy(feature_fraction=0.3, replace=False)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                set_i = set(results[i].feature_indices)
                set_j = set(results[j].feature_indices)
                intersection = set_i & set_j
                self.assertEqual(
                    len(intersection),
                    0,
                    f"Children {i} and {j} have overlapping features: {intersection}",
                )

    def test_feature_mapping_correct(self):
        """feature_mapping correctly maps child indices to parent indices."""
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            self.assertIsNotNone(result.feature_mapping)
            for i, idx in enumerate(result.feature_indices):
                self.assertEqual(result.feature_mapping[i], int(idx))

    def test_data_dimensions_correct(self):
        """Sampled data has correct dimensions."""
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            self.assertEqual(result.data.shape[0], len(self.labels))
            self.assertEqual(result.data.shape[1], len(result.feature_indices))
            self.assertIsNone(result.instance_indices)

    def test_invalid_feature_fraction_raises(self):
        """feature_fraction <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            FeatureSamplingStrategy(feature_fraction=0.0)
        with self.assertRaises(ValueError):
            FeatureSamplingStrategy(feature_fraction=-1.0)


class TestInstanceSamplingStrategy(unittest.TestCase):
    """Tests for InstanceSamplingStrategy."""

    def setUp(self):
        np.random.seed(42)
        self.data = np.random.rand(100, 20) > 0.5
        self.labels = np.random.randint(0, 2, 100)

    def test_returns_correct_number_of_results(self):
        """sample() returns exactly num_children results."""
        strategy = InstanceSamplingStrategy(sample_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_children=5)
        self.assertEqual(len(results), 5)

    def test_each_result_has_unique_instances(self):
        """Each child has unique instance indices (no duplicates within a child)."""
        strategy = InstanceSamplingStrategy(sample_fraction=1.0, replace=True)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            unique_count = len(np.unique(result.instance_indices))
            self.assertEqual(unique_count, len(result.instance_indices))

    def test_no_overlap_when_replace_false(self):
        """When replace=False, instances don't overlap between children."""
        strategy = InstanceSamplingStrategy(sample_fraction=0.3, replace=False)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                set_i = set(results[i].instance_indices)
                set_j = set(results[j].instance_indices)
                intersection = set_i & set_j
                self.assertEqual(
                    len(intersection),
                    0,
                    f"Children {i} and {j} have overlapping instances: {intersection}",
                )

    def test_data_dimensions_correct(self):
        """Sampled data has correct dimensions."""
        strategy = InstanceSamplingStrategy(sample_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            self.assertEqual(result.data.shape[0], len(result.instance_indices))
            self.assertEqual(result.data.shape[1], 20)
            self.assertEqual(len(result.labels), len(result.instance_indices))
            self.assertIsNone(result.feature_mapping)

    def test_invalid_sample_fraction_raises(self):
        """sample_fraction <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            InstanceSamplingStrategy(sample_fraction=0.0)
        with self.assertRaises(ValueError):
            InstanceSamplingStrategy(sample_fraction=-1.0)


class TestCombinedSamplingStrategy(unittest.TestCase):
    """Tests for CombinedSamplingStrategy."""

    def setUp(self):
        np.random.seed(42)
        self.data = np.random.rand(100, 20) > 0.5
        self.labels = np.random.randint(0, 2, 100)

    def test_returns_correct_number_of_results(self):
        """sample() returns exactly num_children results."""
        strategy = CombinedSamplingStrategy(feature_fraction=1.0, sample_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_children=5)
        self.assertEqual(len(results), 5)

    def test_each_result_has_unique_features(self):
        """Each child has unique feature indices (no duplicates within a child)."""
        strategy = CombinedSamplingStrategy(
            feature_fraction=1.0, sample_fraction=1.0, replace=True
        )
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            unique_count = len(np.unique(result.feature_indices))
            self.assertEqual(unique_count, len(result.feature_indices))

    def test_each_result_has_unique_instances(self):
        """Each child has unique instance indices (no duplicates within a child)."""
        strategy = CombinedSamplingStrategy(
            feature_fraction=1.0, sample_fraction=1.0, replace=True
        )
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            unique_count = len(np.unique(result.instance_indices))
            self.assertEqual(unique_count, len(result.instance_indices))

    def test_no_feature_overlap_when_replace_false(self):
        """When replace=False, features don't overlap between children."""
        strategy = CombinedSamplingStrategy(
            feature_fraction=0.3, sample_fraction=0.3, replace=False
        )
        results = strategy.sample(self.data, self.labels, num_children=3)

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                set_i = set(results[i].feature_indices)
                set_j = set(results[j].feature_indices)
                intersection = set_i & set_j
                self.assertEqual(
                    len(intersection),
                    0,
                    f"Children {i} and {j} have overlapping features: {intersection}",
                )

    def test_no_instance_overlap_when_replace_false(self):
        """When replace=False, instances don't overlap between children."""
        strategy = CombinedSamplingStrategy(
            feature_fraction=0.3, sample_fraction=0.3, replace=False
        )
        results = strategy.sample(self.data, self.labels, num_children=3)

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                set_i = set(results[i].instance_indices)
                set_j = set(results[j].instance_indices)
                intersection = set_i & set_j
                self.assertEqual(
                    len(intersection),
                    0,
                    f"Children {i} and {j} have overlapping instances: {intersection}",
                )

    def test_feature_mapping_correct(self):
        """feature_mapping correctly maps child indices to parent indices."""
        strategy = CombinedSamplingStrategy(feature_fraction=1.0, sample_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            self.assertIsNotNone(result.feature_mapping)
            for i, idx in enumerate(result.feature_indices):
                self.assertEqual(result.feature_mapping[i], int(idx))

    def test_data_dimensions_correct(self):
        """Sampled data has correct dimensions."""
        strategy = CombinedSamplingStrategy(feature_fraction=1.0, sample_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            self.assertEqual(result.data.shape[0], len(result.instance_indices))
            self.assertEqual(result.data.shape[1], len(result.feature_indices))
            self.assertEqual(len(result.labels), len(result.instance_indices))

    def test_invalid_fractions_raise(self):
        """Invalid fractions raise ValueError."""
        with self.assertRaises(ValueError):
            CombinedSamplingStrategy(feature_fraction=0.0)
        with self.assertRaises(ValueError):
            CombinedSamplingStrategy(sample_fraction=-1.0)


class TestSamplingRandomized(unittest.TestCase):
    """Randomized tests for sampling strategies.

    These tests verify behavior across multiple random configurations.
    """

    def _run_randomized_test(self, test_fn, iterations=50):
        """Run a test multiple times with random parameters."""
        for _ in range(iterations):
            test_fn()

    def test_feature_sampling_returns_correct_number_of_results(self):
        """FeatureSamplingStrategy sample() returns exactly num_children results."""

        def check():
            num_features = np.random.randint(4, 51)
            num_instances = np.random.randint(4, 101)
            num_children = np.random.randint(1, 11)
            replace = bool(np.random.choice([True, False]))
            feature_fraction = float(np.random.uniform(0.5, 1.0))

            data = np.random.rand(num_instances, num_features) > 0.5
            labels = np.random.randint(0, 2, num_instances)

            strategy = FeatureSamplingStrategy(
                feature_fraction=feature_fraction, replace=replace
            )
            results = strategy.sample(data, labels, num_children=int(num_children))

            self.assertEqual(len(results), num_children)

        self._run_randomized_test(check)

    def test_instance_sampling_returns_correct_number_of_results(self):
        """InstanceSamplingStrategy sample() returns exactly num_children results."""

        def check():
            num_features = np.random.randint(4, 51)
            num_instances = np.random.randint(4, 101)
            num_children = np.random.randint(1, 11)
            replace = bool(np.random.choice([True, False]))
            sample_fraction = float(np.random.uniform(0.5, 1.0))

            data = np.random.rand(num_instances, num_features) > 0.5
            labels = np.random.randint(0, 2, num_instances)

            strategy = InstanceSamplingStrategy(
                sample_fraction=sample_fraction, replace=replace
            )
            results = strategy.sample(data, labels, num_children=int(num_children))

            self.assertEqual(len(results), num_children)

        self._run_randomized_test(check)

    def test_combined_sampling_returns_correct_number_of_results(self):
        """CombinedSamplingStrategy sample() returns exactly num_children results."""

        def check():
            num_features = np.random.randint(4, 51)
            num_instances = np.random.randint(4, 101)
            num_children = np.random.randint(1, 11)
            replace = bool(np.random.choice([True, False]))
            feature_fraction = float(np.random.uniform(0.5, 1.0))
            sample_fraction = float(np.random.uniform(0.5, 1.0))

            data = np.random.rand(num_instances, num_features) > 0.5
            labels = np.random.randint(0, 2, num_instances)

            strategy = CombinedSamplingStrategy(
                feature_fraction=feature_fraction,
                sample_fraction=sample_fraction,
                replace=replace,
            )
            results = strategy.sample(data, labels, num_children=int(num_children))

            self.assertEqual(len(results), num_children)

        self._run_randomized_test(check)

    def test_feature_sampling_feature_mapping_correct(self):
        """FeatureSamplingStrategy feature_mapping correctly maps child to parent indices."""

        def check():
            num_features = np.random.randint(4, 51)
            num_instances = np.random.randint(4, 101)
            num_children = np.random.randint(1, 11)
            replace = bool(np.random.choice([True, False]))
            feature_fraction = float(np.random.uniform(0.5, 1.0))

            data = np.random.rand(num_instances, num_features) > 0.5
            labels = np.random.randint(0, 2, num_instances)

            strategy = FeatureSamplingStrategy(
                feature_fraction=feature_fraction, replace=replace
            )
            results = strategy.sample(data, labels, num_children=int(num_children))

            for result in results:
                self.assertIsNotNone(result.feature_mapping)
                expected_keys = set(range(len(result.feature_indices)))
                self.assertEqual(set(result.feature_mapping.keys()), expected_keys)
                for i, idx in enumerate(result.feature_indices):
                    self.assertEqual(result.feature_mapping[i], int(idx))

        self._run_randomized_test(check)

    def test_combined_sampling_feature_mapping_correct(self):
        """CombinedSamplingStrategy feature_mapping correctly maps child to parent indices."""

        def check():
            num_features = np.random.randint(4, 51)
            num_instances = np.random.randint(4, 101)
            num_children = np.random.randint(1, 11)
            replace = bool(np.random.choice([True, False]))
            feature_fraction = float(np.random.uniform(0.5, 1.0))
            sample_fraction = float(np.random.uniform(0.5, 1.0))

            data = np.random.rand(num_instances, num_features) > 0.5
            labels = np.random.randint(0, 2, num_instances)

            strategy = CombinedSamplingStrategy(
                feature_fraction=feature_fraction,
                sample_fraction=sample_fraction,
                replace=replace,
            )
            results = strategy.sample(data, labels, num_children=int(num_children))

            for result in results:
                self.assertIsNotNone(result.feature_mapping)
                expected_keys = set(range(len(result.feature_indices)))
                self.assertEqual(set(result.feature_mapping.keys()), expected_keys)
                for i, idx in enumerate(result.feature_indices):
                    self.assertEqual(result.feature_mapping[i], int(idx))

        self._run_randomized_test(check)


if __name__ == "__main__":
    unittest.main()
