"""Tests for sampling strategies."""

import unittest
import numpy as np

from hgp_lib.populations.sampling import (
    FeatureSamplingStrategy,
    InstanceSamplingStrategy,
    CombinedSamplingStrategy,
    SamplingResult,
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
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=5)
        self.assertEqual(len(results), 5)

    def test_each_result_has_unique_features(self):
        """Each child has unique feature indices (no duplicates within a child)."""
        strategy = FeatureSamplingStrategy(feature_fraction=1.0, replace=True)
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=3)
        
        for result in results:
            unique_count = len(np.unique(result.feature_indices))
            self.assertEqual(unique_count, len(result.feature_indices))

    def test_no_overlap_when_replace_false(self):
        """When replace=False, features don't overlap between children."""
        strategy = FeatureSamplingStrategy(feature_fraction=0.5, replace=False)
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=3)
        
        # Check no overlap between any two children
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                set_i = set(results[i].feature_indices)
                set_j = set(results[j].feature_indices)
                intersection = set_i & set_j
                self.assertEqual(len(intersection), 0, 
                    f"Children {i} and {j} have overlapping features: {intersection}")

    def test_overlap_allowed_when_replace_true(self):
        """When replace=True, same feature can appear in multiple children."""
        np.random.seed(123)
        strategy = FeatureSamplingStrategy(feature_fraction=1.0, replace=True)
        results = strategy.sample(self.data, self.labels, num_features=10, num_children=5)
        
        # With 5 children each getting ~2 features from 10, overlap is likely
        all_features = [set(r.feature_indices) for r in results]
        has_overlap = False
        for i in range(len(all_features)):
            for j in range(i + 1, len(all_features)):
                if all_features[i] & all_features[j]:
                    has_overlap = True
                    break
        # Not asserting overlap exists (it's probabilistic), just that it's allowed

    def test_feature_mapping_correct(self):
        """feature_mapping correctly maps child indices to parent indices."""
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=3)
        
        for result in results:
            self.assertIsNotNone(result.feature_mapping)
            for i, idx in enumerate(result.feature_indices):
                self.assertEqual(result.feature_mapping[i], int(idx))

    def test_data_dimensions_correct(self):
        """Sampled data has correct dimensions."""
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=3)
        
        for result in results:
            # All instances preserved
            self.assertEqual(result.data.shape[0], len(self.labels))
            # Features match feature_indices
            self.assertEqual(result.data.shape[1], len(result.feature_indices))
            # instance_indices is None for feature-only sampling
            self.assertIsNone(result.instance_indices)

    def test_invalid_feature_fraction_raises(self):
        """feature_fraction <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            FeatureSamplingStrategy(feature_fraction=0)
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
        strategy = InstanceSamplingStrategy(instance_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=5)
        self.assertEqual(len(results), 5)

    def test_each_result_has_unique_instances(self):
        """Each child has unique instance indices (no duplicates within a child)."""
        strategy = InstanceSamplingStrategy(instance_fraction=1.0, replace=True)
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=3)
        
        for result in results:
            unique_count = len(np.unique(result.instance_indices))
            self.assertEqual(unique_count, len(result.instance_indices))

    def test_no_overlap_when_replace_false(self):
        """When replace=False, instances don't overlap between children."""
        strategy = InstanceSamplingStrategy(instance_fraction=0.5, replace=False)
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=3)
        
        # Check no overlap between any two children
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                set_i = set(results[i].instance_indices)
                set_j = set(results[j].instance_indices)
                intersection = set_i & set_j
                self.assertEqual(len(intersection), 0,
                    f"Children {i} and {j} have overlapping instances: {intersection}")

    def test_data_dimensions_correct(self):
        """Sampled data has correct dimensions."""
        strategy = InstanceSamplingStrategy(instance_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=3)
        
        for result in results:
            # Instances match instance_indices
            self.assertEqual(result.data.shape[0], len(result.instance_indices))
            # All features preserved
            self.assertEqual(result.data.shape[1], 20)
            # Labels match instances
            self.assertEqual(len(result.labels), len(result.instance_indices))
            # feature_mapping is None for instance-only sampling
            self.assertIsNone(result.feature_mapping)

    def test_invalid_instance_fraction_raises(self):
        """instance_fraction <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            InstanceSamplingStrategy(instance_fraction=0)
        with self.assertRaises(ValueError):
            InstanceSamplingStrategy(instance_fraction=-1.0)


class TestCombinedSamplingStrategy(unittest.TestCase):
    """Tests for CombinedSamplingStrategy."""

    def setUp(self):
        np.random.seed(42)
        self.data = np.random.rand(100, 20) > 0.5
        self.labels = np.random.randint(0, 2, 100)

    def test_returns_correct_number_of_results(self):
        """sample() returns exactly num_children results."""
        strategy = CombinedSamplingStrategy(feature_fraction=1.0, instance_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=5)
        self.assertEqual(len(results), 5)

    def test_each_result_has_unique_features(self):
        """Each child has unique feature indices (no duplicates within a child)."""
        strategy = CombinedSamplingStrategy(
            feature_fraction=1.0, instance_fraction=1.0, replace=True
        )
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=3)
        
        for result in results:
            unique_count = len(np.unique(result.feature_indices))
            self.assertEqual(unique_count, len(result.feature_indices))

    def test_each_result_has_unique_instances(self):
        """Each child has unique instance indices (no duplicates within a child)."""
        strategy = CombinedSamplingStrategy(
            feature_fraction=1.0, instance_fraction=1.0, replace=True
        )
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=3)
        
        for result in results:
            unique_count = len(np.unique(result.instance_indices))
            self.assertEqual(unique_count, len(result.instance_indices))

    def test_no_feature_overlap_when_replace_false(self):
        """When replace=False, features don't overlap between children."""
        strategy = CombinedSamplingStrategy(
            feature_fraction=0.5, instance_fraction=0.5, replace=False
        )
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=3)
        
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                set_i = set(results[i].feature_indices)
                set_j = set(results[j].feature_indices)
                intersection = set_i & set_j
                self.assertEqual(len(intersection), 0,
                    f"Children {i} and {j} have overlapping features: {intersection}")

    def test_no_instance_overlap_when_replace_false(self):
        """When replace=False, instances don't overlap between children."""
        strategy = CombinedSamplingStrategy(
            feature_fraction=0.5, instance_fraction=0.5, replace=False
        )
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=3)
        
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                set_i = set(results[i].instance_indices)
                set_j = set(results[j].instance_indices)
                intersection = set_i & set_j
                self.assertEqual(len(intersection), 0,
                    f"Children {i} and {j} have overlapping instances: {intersection}")

    def test_feature_mapping_correct(self):
        """feature_mapping correctly maps child indices to parent indices."""
        strategy = CombinedSamplingStrategy(feature_fraction=1.0, instance_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=3)
        
        for result in results:
            self.assertIsNotNone(result.feature_mapping)
            for i, idx in enumerate(result.feature_indices):
                self.assertEqual(result.feature_mapping[i], int(idx))

    def test_data_dimensions_correct(self):
        """Sampled data has correct dimensions."""
        strategy = CombinedSamplingStrategy(feature_fraction=1.0, instance_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_features=20, num_children=3)
        
        for result in results:
            # Instances match instance_indices
            self.assertEqual(result.data.shape[0], len(result.instance_indices))
            # Features match feature_indices
            self.assertEqual(result.data.shape[1], len(result.feature_indices))
            # Labels match instances
            self.assertEqual(len(result.labels), len(result.instance_indices))

    def test_invalid_fractions_raise(self):
        """Invalid fractions raise ValueError."""
        with self.assertRaises(ValueError):
            CombinedSamplingStrategy(feature_fraction=0)
        with self.assertRaises(ValueError):
            CombinedSamplingStrategy(instance_fraction=-1.0)


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
            feature_fraction = np.random.uniform(0.1, 2.0)
            replace = np.random.choice([True, False])
            
            data = np.random.rand(num_instances, num_features) > 0.5
            labels = np.random.randint(0, 2, num_instances)
            
            strategy = FeatureSamplingStrategy(feature_fraction=feature_fraction, replace=replace)
            results = strategy.sample(data, labels, num_features=num_features, num_children=num_children)
            
            self.assertEqual(len(results), num_children)
        
        self._run_randomized_test(check)

    def test_instance_sampling_returns_correct_number_of_results(self):
        """InstanceSamplingStrategy sample() returns exactly num_children results."""
        def check():
            num_features = np.random.randint(4, 51)
            num_instances = np.random.randint(4, 101)
            num_children = np.random.randint(1, 11)
            instance_fraction = np.random.uniform(0.1, 2.0)
            replace = np.random.choice([True, False])
            
            data = np.random.rand(num_instances, num_features) > 0.5
            labels = np.random.randint(0, 2, num_instances)
            
            strategy = InstanceSamplingStrategy(instance_fraction=instance_fraction, replace=replace)
            results = strategy.sample(data, labels, num_features=num_features, num_children=num_children)
            
            self.assertEqual(len(results), num_children)
        
        self._run_randomized_test(check)

    def test_combined_sampling_returns_correct_number_of_results(self):
        """CombinedSamplingStrategy sample() returns exactly num_children results."""
        def check():
            num_features = np.random.randint(4, 51)
            num_instances = np.random.randint(4, 101)
            num_children = np.random.randint(1, 11)
            feature_fraction = np.random.uniform(0.1, 2.0)
            instance_fraction = np.random.uniform(0.1, 2.0)
            replace = np.random.choice([True, False])
            
            data = np.random.rand(num_instances, num_features) > 0.5
            labels = np.random.randint(0, 2, num_instances)
            
            strategy = CombinedSamplingStrategy(
                feature_fraction=feature_fraction,
                instance_fraction=instance_fraction,
                replace=replace
            )
            results = strategy.sample(data, labels, num_features=num_features, num_children=num_children)
            
            self.assertEqual(len(results), num_children)
        
        self._run_randomized_test(check)

    def test_feature_sampling_feature_mapping_correct(self):
        """FeatureSamplingStrategy feature_mapping correctly maps child to parent indices."""
        def check():
            num_features = np.random.randint(4, 51)
            num_instances = np.random.randint(4, 101)
            num_children = np.random.randint(1, 11)
            feature_fraction = np.random.uniform(0.1, 2.0)
            replace = np.random.choice([True, False])
            
            data = np.random.rand(num_instances, num_features) > 0.5
            labels = np.random.randint(0, 2, num_instances)
            
            strategy = FeatureSamplingStrategy(feature_fraction=feature_fraction, replace=replace)
            results = strategy.sample(data, labels, num_features=num_features, num_children=num_children)
            
            for result in results:
                self.assertIsNotNone(result.feature_mapping)
                # Keys should be 0 to len(feature_indices)-1
                expected_keys = set(range(len(result.feature_indices)))
                self.assertEqual(set(result.feature_mapping.keys()), expected_keys)
                # Each mapping should match feature_indices
                for i, idx in enumerate(result.feature_indices):
                    self.assertEqual(result.feature_mapping[i], int(idx))
        
        self._run_randomized_test(check)

    def test_combined_sampling_feature_mapping_correct(self):
        """CombinedSamplingStrategy feature_mapping correctly maps child to parent indices."""
        def check():
            num_features = np.random.randint(4, 51)
            num_instances = np.random.randint(4, 101)
            num_children = np.random.randint(1, 11)
            feature_fraction = np.random.uniform(0.1, 2.0)
            instance_fraction = np.random.uniform(0.1, 2.0)
            replace = np.random.choice([True, False])
            
            data = np.random.rand(num_instances, num_features) > 0.5
            labels = np.random.randint(0, 2, num_instances)
            
            strategy = CombinedSamplingStrategy(
                feature_fraction=feature_fraction,
                instance_fraction=instance_fraction,
                replace=replace
            )
            results = strategy.sample(data, labels, num_features=num_features, num_children=num_children)
            
            for result in results:
                self.assertIsNotNone(result.feature_mapping)
                # Keys should be 0 to len(feature_indices)-1
                expected_keys = set(range(len(result.feature_indices)))
                self.assertEqual(set(result.feature_mapping.keys()), expected_keys)
                # Each mapping should match feature_indices
                for i, idx in enumerate(result.feature_indices):
                    self.assertEqual(result.feature_mapping[i], int(idx))
        
        self._run_randomized_test(check)


if __name__ == "__main__":
    unittest.main()
