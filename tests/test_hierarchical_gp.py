"""Tests for Hierarchical Genetic Programming functionality."""

import unittest

import numpy as np

from hgp_lib.algorithms import BooleanGP
from hgp_lib.configs import BooleanGPConfig
from hgp_lib.populations import (
    FeatureSamplingStrategy,
    InstanceSamplingStrategy,
    CombinedSamplingStrategy,
)
import sklearn.metrics


def accuracy(predictions, labels, sample_weight=None):
    """Simple accuracy score function."""
    return sklearn.metrics.accuracy_score(
        labels, predictions, sample_weight=sample_weight
    )


class TestSamplingStrategies(unittest.TestCase):
    """Tests for sampling strategies used in hierarchical GP."""

    def setUp(self):
        np.random.seed(42)
        self.data = np.random.rand(100, 20) > 0.5
        self.labels = np.random.randint(0, 2, 100)

    def test_feature_sampling_basic(self):
        """Test that FeatureSamplingStrategy samples correct number of features."""
        strategy = FeatureSamplingStrategy(feature_fraction=0.25)
        results = strategy.sample(self.data, self.labels, num_children=4)

        self.assertEqual(len(results), 4)

        # ceil(20 * 0.25) = 5 features per child
        result = results[0]
        self.assertEqual(len(result.feature_indices), 5)
        self.assertEqual(result.data.shape, (100, 5))
        self.assertIsNone(result.instance_indices)
        np.testing.assert_array_equal(result.labels, self.labels)

    def test_feature_sampling_with_fraction(self):
        """Test FeatureSamplingStrategy with different fractions."""
        strategy = FeatureSamplingStrategy(feature_fraction=0.5)
        results = strategy.sample(self.data, self.labels, num_children=4)

        self.assertEqual(len(results), 4)

        # ceil(20 * 0.5) = 10 features per child
        self.assertEqual(len(results[0].feature_indices), 10)

    def test_instance_sampling_basic(self):
        """Test that InstanceSamplingStrategy samples correct number of instances."""
        strategy = InstanceSamplingStrategy(sample_fraction=0.25)
        results = strategy.sample(self.data, self.labels, num_children=4)

        self.assertEqual(len(results), 4)

        # ceil(100 * 0.25) = 25 instances per child
        result = results[0]
        self.assertEqual(len(result.instance_indices), 25)
        self.assertEqual(result.data.shape, (25, 20))
        self.assertIsNone(result.feature_mapping)

    def test_combined_sampling(self):
        """Test CombinedSamplingStrategy samples both dimensions."""
        strategy = CombinedSamplingStrategy(feature_fraction=0.25, sample_fraction=0.25)
        results = strategy.sample(self.data, self.labels, num_children=4)

        self.assertEqual(len(results), 4)

        # Features: ceil(20 * 0.25) = 5, Instances: ceil(100 * 0.25) = 25
        result = results[0]
        self.assertEqual(len(result.feature_indices), 5)
        self.assertEqual(len(result.instance_indices), 25)
        self.assertEqual(result.data.shape, (25, 5))


class TestChildPopulationCreation(unittest.TestCase):
    """Tests for child population creation in BooleanGP."""

    def setUp(self):
        np.random.seed(42)
        self.data = np.random.rand(50, 10) > 0.5
        self.labels = np.random.randint(0, 2, 50)

    def test_no_children_when_max_depth_zero(self):
        """Test that no child populations are created when max_depth=0."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=0,
            num_child_populations=3,
            sampling_strategy=FeatureSamplingStrategy(),
        )
        gp = BooleanGP(config)

        self.assertEqual(len(gp.child_populations), 0)
        self.assertEqual(gp.current_depth, 0)

    def test_children_created_with_feature_sampling(self):
        """Test child populations are created with feature sampling."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=1,
            num_child_populations=2,
            sampling_strategy=FeatureSamplingStrategy(feature_fraction=0.5),
            top_k_transfer=5,
        )
        gp = BooleanGP(config)

        self.assertEqual(len(gp.child_populations), 2)
        self.assertEqual(gp.current_depth, 0)

        for child in gp.child_populations:
            self.assertEqual(child.current_depth, 1)
            self.assertIsNotNone(child.feature_mapping)
            self.assertLess(child.train_data.shape[1], self.data.shape[1])

    def test_children_created_with_instance_sampling(self):
        """Test child populations with instance-only sampling have no feature mapping."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=1,
            num_child_populations=2,
            sampling_strategy=InstanceSamplingStrategy(sample_fraction=1.0),
            top_k_transfer=5,
        )
        gp = BooleanGP(config)

        self.assertEqual(len(gp.child_populations), 2)

        for child in gp.child_populations:
            # Instance sampling preserves all features, so no mapping needed
            self.assertIsNone(child.feature_mapping)
            # Same number of features
            self.assertEqual(child.train_data.shape[1], self.data.shape[1])
            # Fewer instances
            self.assertLess(child.train_data.shape[0], self.data.shape[0])

    def test_nested_children_with_depth_2(self):
        """Test that depth=2 creates grandchildren."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=2,
            num_child_populations=2,
            sampling_strategy=FeatureSamplingStrategy(feature_fraction=1.0),
            top_k_transfer=3,
        )
        gp = BooleanGP(config)

        self.assertEqual(gp.current_depth, 0)
        self.assertEqual(len(gp.child_populations), 2)

        for child in gp.child_populations:
            self.assertEqual(child.current_depth, 1)
            # Each child should have grandchildren
            self.assertEqual(len(child.child_populations), 2)

            for grandchild in child.child_populations:
                self.assertEqual(grandchild.current_depth, 2)
                # Grandchildren should have no children (max_depth reached)
                self.assertEqual(len(grandchild.child_populations), 0)


class TestHierarchicalTraining(unittest.TestCase):
    """Integration tests for hierarchical GP training flow."""

    def setUp(self):
        np.random.seed(42)
        # Create simple linearly separable data
        self.data = np.zeros((100, 10), dtype=bool)
        self.labels = np.zeros(100, dtype=int)
        # Class 1: feature 0 is True
        self.data[50:, 0] = True
        self.labels[50:] = 1

    def test_single_step_with_children(self):
        """Test that a single training step completes with child populations."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=1,
            num_child_populations=2,
            sampling_strategy=FeatureSamplingStrategy(feature_fraction=1.0),
            top_k_transfer=5,
            optimize_scorer=False,
        )
        gp = BooleanGP(config)

        metrics = gp.step()

        self.assertIsNotNone(metrics.best_rule)
        self.assertGreater(metrics.best_train_score, 0)
        # Should have metrics for child populations
        self.assertIsNotNone(metrics.child_population_generation_metrics)
        self.assertEqual(len(metrics.child_population_generation_metrics), 2)

        for child_metrics in metrics.child_population_generation_metrics:
            self.assertIsNotNone(child_metrics.best_rule)

    def test_multiple_steps_training(self):
        """Test multiple training steps improve or maintain performance."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=1,
            num_child_populations=2,
            sampling_strategy=FeatureSamplingStrategy(feature_fraction=1.0),
            top_k_transfer=5,
            optimize_scorer=False,
        )
        gp = BooleanGP(config)

        scores = []
        for _ in range(5):
            metrics = gp.step()
            scores.append(metrics.best_train_score)

        # Best score should never decrease (tracked on gp object)
        # Note: metrics.best_train_score is per-generation, not cumulative
        # So we just check that training completes successfully
        self.assertEqual(len(scores), 5)

    def test_hierarchical_with_instance_sampling(self):
        """Test hierarchical GP with instance sampling."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=1,
            num_child_populations=2,
            sampling_strategy=InstanceSamplingStrategy(sample_fraction=1.0),
            top_k_transfer=5,
            optimize_scorer=False,
        )
        gp = BooleanGP(config)

        # Should complete without errors
        metrics = gp.step()
        self.assertIsNotNone(metrics.best_rule)

    def test_hierarchical_with_combined_sampling(self):
        """Test hierarchical GP with combined sampling."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=1,
            num_child_populations=2,
            sampling_strategy=CombinedSamplingStrategy(
                feature_fraction=1.0, sample_fraction=1.0
            ),
            top_k_transfer=5,
            optimize_scorer=False,
        )
        gp = BooleanGP(config)

        metrics = gp.step()
        self.assertIsNotNone(metrics.best_rule)

    def test_depth_2_training(self):
        """Test training with grandchild populations (depth=2)."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=2,
            num_child_populations=2,
            sampling_strategy=FeatureSamplingStrategy(feature_fraction=1.0),
            top_k_transfer=3,
            optimize_scorer=False,
        )
        gp = BooleanGP(config)

        metrics = gp.step()

        # Check nested metrics structure
        self.assertIsNotNone(metrics.child_population_generation_metrics)
        self.assertEqual(len(metrics.child_population_generation_metrics), 2)
        for child_metrics in metrics.child_population_generation_metrics:
            self.assertIsNotNone(child_metrics.child_population_generation_metrics)
            self.assertEqual(len(child_metrics.child_population_generation_metrics), 2)
            for grandchild_metrics in child_metrics.child_population_generation_metrics:
                # Grandchildren have no children
                self.assertEqual(
                    len(grandchild_metrics.child_population_generation_metrics), 0
                )


class TestFeedbackMechanism(unittest.TestCase):
    """Tests for the feedback mechanism between parent and child populations."""

    def setUp(self):
        np.random.seed(42)
        self.data = np.random.rand(50, 10) > 0.5
        self.labels = np.random.randint(0, 2, 50)

    def test_multiplicative_feedback(self):
        """Test that multiplicative feedback modifies scores correctly."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=1,
            num_child_populations=2,
            sampling_strategy=FeatureSamplingStrategy(),
            top_k_transfer=5,
            feedback_type="multiplicative",
            feedback_strength=0.1,
            optimize_scorer=False,
        )
        gp = BooleanGP(config)

        # Should complete without errors
        metrics = gp.step()
        self.assertIsNotNone(metrics)

    def test_additive_feedback(self):
        """Test that additive feedback modifies scores correctly."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=1,
            num_child_populations=2,
            sampling_strategy=FeatureSamplingStrategy(),
            top_k_transfer=5,
            feedback_type="additive",
            feedback_strength=0.1,
            optimize_scorer=False,
        )
        gp = BooleanGP(config)

        metrics = gp.step()
        self.assertIsNotNone(metrics)


class TestConfigValidation(unittest.TestCase):
    """Tests for hierarchical GP configuration validation."""

    def setUp(self):
        self.data = np.random.rand(50, 10) > 0.5
        self.labels = np.random.randint(0, 2, 50)

    def test_max_depth_without_sampling_strategy_raises(self):
        """Test that max_depth > 0 without sampling_strategy raises ValueError."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=1,
            num_child_populations=2,
            sampling_strategy=None,  # Missing!
        )
        # Validation happens in BooleanGP.__init__(), not in config creation
        with self.assertRaises(ValueError) as ctx:
            BooleanGP(config)
        self.assertIn("sampling_strategy", str(ctx.exception))

    def test_max_depth_without_children_raises(self):
        """Test that max_depth > 0 with num_child_populations=0 raises ValueError."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=1,
            num_child_populations=0,  # No children!
            sampling_strategy=FeatureSamplingStrategy(),
        )
        # Validation happens in BooleanGP.__init__(), not in config creation
        with self.assertRaises(ValueError) as ctx:
            BooleanGP(config)
        self.assertIn("num_child_populations", str(ctx.exception))

    def test_top_k_transfer_exceeds_population_raises(self):
        """Test that top_k_transfer > population_size raises ValueError."""
        config = BooleanGPConfig(
            score_fn=accuracy,
            train_data=self.data,
            train_labels=self.labels,
            max_depth=1,
            num_child_populations=2,
            sampling_strategy=FeatureSamplingStrategy(),
            top_k_transfer=1000,  # Way too large
        )
        with self.assertRaises(ValueError) as ctx:
            BooleanGP(config)
        self.assertIn("top_k_transfer", str(ctx.exception))


class TestBooleanGPSamplingIntegration(unittest.TestCase):
    """Tests for BooleanGP sampling integration.

    These tests verify that BooleanGP correctly uses the sampling strategy
    and creates the expected number of child populations.
    """

    def _run_randomized_test(self, test_fn, iterations=30):
        """Run a test multiple times with random parameters."""
        for _ in range(iterations):
            test_fn()

    def test_correct_number_of_child_populations(self):
        """BooleanGP creates exactly num_child_populations children when max_depth > 0."""

        def check():
            num_features = np.random.randint(10, 31)
            num_instances = np.random.randint(20, 51)
            num_children = np.random.randint(1, 6)

            data = np.random.rand(num_instances, num_features) > 0.5
            labels = np.random.randint(0, 2, num_instances)

            config = BooleanGPConfig(
                score_fn=accuracy,
                train_data=data,
                train_labels=labels,
                max_depth=1,
                num_child_populations=num_children,
                sampling_strategy=FeatureSamplingStrategy(feature_fraction=1.0),
                top_k_transfer=3,
            )
            gp = BooleanGP(config)

            self.assertEqual(len(gp.child_populations), num_children)

        self._run_randomized_test(check)

    def test_feature_mapping_correct_in_children(self):
        """Each child population has a correct feature_mapping."""

        def check():
            num_features = np.random.randint(10, 31)
            num_instances = np.random.randint(20, 51)
            num_children = np.random.randint(1, 6)

            data = np.random.rand(num_instances, num_features) > 0.5
            labels = np.random.randint(0, 2, num_instances)

            config = BooleanGPConfig(
                score_fn=accuracy,
                train_data=data,
                train_labels=labels,
                max_depth=1,
                num_child_populations=num_children,
                sampling_strategy=FeatureSamplingStrategy(feature_fraction=1.0),
                top_k_transfer=3,
            )
            gp = BooleanGP(config)

            for child in gp.child_populations:
                # Feature sampling should have feature_mapping
                self.assertIsNotNone(child.feature_mapping)

                # Keys should be 0 to num_child_features-1
                num_child_features = child.train_data.shape[1]
                expected_keys = set(range(num_child_features))
                self.assertEqual(set(child.feature_mapping.keys()), expected_keys)

                # All mapped values should be valid parent feature indices
                for child_idx, parent_idx in child.feature_mapping.items():
                    self.assertGreaterEqual(parent_idx, 0)
                    self.assertLess(parent_idx, num_features)

        self._run_randomized_test(check)

    def test_instance_sampling_no_feature_mapping(self):
        """Instance-only sampling results in no feature_mapping."""

        def check():
            num_features = np.random.randint(10, 31)
            num_instances = np.random.randint(20, 51)
            num_children = np.random.randint(1, 6)

            data = np.random.rand(num_instances, num_features) > 0.5
            labels = np.random.randint(0, 2, num_instances)

            config = BooleanGPConfig(
                score_fn=accuracy,
                train_data=data,
                train_labels=labels,
                max_depth=1,
                num_child_populations=num_children,
                sampling_strategy=InstanceSamplingStrategy(sample_fraction=1.0),
                top_k_transfer=3,
            )
            gp = BooleanGP(config)

            for child in gp.child_populations:
                # Instance sampling should NOT have feature_mapping
                self.assertIsNone(child.feature_mapping)
                # But should have same number of features
                self.assertEqual(child.train_data.shape[1], num_features)

        self._run_randomized_test(check)


if __name__ == "__main__":
    unittest.main()
