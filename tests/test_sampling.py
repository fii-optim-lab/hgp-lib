"""Tests for sampling strategies."""

import unittest

import numpy as np

from hgp_lib.populations.sampling import (
    CombinedSamplingStrategy,
    FeatureSamplingStrategy,
    InstanceSamplingStrategy,
)


def identifiable_data(num_instances: int, num_features: int) -> np.ndarray:
    """Boolean data whose every row encodes its own row index in binary.

    Rows are therefore unique and their original index can be recovered with
    `recover_row_indices`, which lets tests check which instances a child
    received without the strategies having to report indices.

    Requires `2 ** num_features > num_instances`.
    """
    assert 2**num_features > num_instances, "not enough features to encode row indices"
    return ((np.arange(num_instances)[:, None] >> np.arange(num_features)) & 1).astype(
        bool
    )


def recover_row_indices(data: np.ndarray) -> np.ndarray:
    """Recover the original row indices from data built by `identifiable_data`."""
    return data.astype(int) @ (1 << np.arange(data.shape[1]))


class TestIndexAllocation(unittest.TestCase):
    """Tests for `SamplingStrategy.allocate_indices_to_children`.

    This helper owns the overlap and uniqueness behaviour that `replace`
    controls, so the invariants are asserted here rather than through the
    sampled data of each strategy.
    """

    def setUp(self):
        np.random.seed(42)

    @staticmethod
    def _allocate(k, n, num_children, replace):
        strategy = FeatureSamplingStrategy(replace=replace)
        return strategy.allocate_indices_to_children(k, n, num_children)

    def test_returns_one_allocation_per_child(self):
        for k, n, num_children, replace in [
            (10, 10, 3, False),  # k >= n
            (3, 10, 3, False),  # partitioned
            (3, 10, 3, True),  # independent samples
            (4, 10, 3, False),  # partition impossible, independent samples
        ]:
            with self.subTest(k=k, n=n, num_children=num_children, replace=replace):
                allocation = self._allocate(k, n, num_children, replace)
                self.assertEqual(len(allocation), num_children)

    def test_all_indices_given_when_k_at_least_n(self):
        """With k >= n every child receives all n indices."""
        for k in (10, 15):
            with self.subTest(k=k):
                allocation = self._allocate(k, n=10, num_children=3, replace=False)
                for indices in allocation:
                    np.testing.assert_array_equal(indices, np.arange(10))

    def test_partitions_without_overlap(self):
        """With replace=False and k * num_children <= n children are disjoint."""
        allocation = self._allocate(k=3, n=10, num_children=3, replace=False)

        seen = set()
        for indices in allocation:
            self.assertEqual(len(indices), 3)
            current = set(indices.tolist())
            self.assertEqual(len(current & seen), 0, "children share an index")
            seen |= current
        self.assertEqual(len(seen), 9)

    def test_independent_samples_when_replace_true(self):
        """With replace=True children are sampled independently of each other."""
        allocation = self._allocate(k=3, n=10, num_children=3, replace=True)
        for indices in allocation:
            self.assertEqual(len(indices), 3)

    def test_independent_samples_when_partition_impossible(self):
        """With replace=False but k * num_children > n each child is sampled on its own."""
        allocation = self._allocate(k=4, n=10, num_children=3, replace=False)
        for indices in allocation:
            self.assertEqual(len(indices), 4)

    def test_indices_are_unique_within_each_child_and_in_range(self):
        for k, n, num_children, replace in [
            (10, 10, 3, False),
            (3, 10, 3, False),
            (3, 10, 3, True),
            (4, 10, 3, False),
            (7, 20, 4, True),
        ]:
            with self.subTest(k=k, n=n, num_children=num_children, replace=replace):
                allocation = self._allocate(k, n, num_children, replace)
                for indices in allocation:
                    self.assertEqual(len(np.unique(indices)), len(indices))
                    self.assertTrue(((indices >= 0) & (indices < n)).all())


class SamplingAssertions(unittest.TestCase):
    """Shared assertions about a SamplingResult, all derived from its data."""

    def assert_feature_mapping_well_formed(self, result, num_parent_features):
        """feature_mapping keys cover the child columns and values are valid parents."""
        self.assertIsNotNone(result.feature_mapping)
        self.assertEqual(
            set(result.feature_mapping.keys()), set(range(result.data.shape[1]))
        )
        values = list(result.feature_mapping.values())
        self.assertEqual(len(set(values)), len(values), "duplicate parent features")
        for value in values:
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)
            self.assertLess(value, num_parent_features)

    def assert_columns_match_mapping(self, result, parent_data, rows=slice(None)):
        """Each child column holds the parent column that feature_mapping names.

        This is what makes the mapping trustworthy: the columns actually kept in
        `data` and the columns recorded in `feature_mapping` must agree, since
        crossover relies on the mapping to translate rules back to the parent.
        """
        for child_col, parent_col in result.feature_mapping.items():
            np.testing.assert_array_equal(
                result.data[:, child_col],
                parent_data[rows, parent_col],
                f"child column {child_col} is not parent column {parent_col}",
            )

    def assert_rows_exist_in_parent(self, result, parent_data, parent_labels):
        """Every (row, label) pair of the child occurs in the parent's projection."""
        if result.feature_mapping is None:
            projection = parent_data
        else:
            projection = parent_data[:, list(result.feature_mapping.values())]

        for row, label in zip(result.data, result.labels):
            candidates = np.flatnonzero((projection == row).all(axis=1))
            self.assertGreater(candidates.size, 0, "child row is not a parent row")
            self.assertIn(label, parent_labels[candidates])


class TestFeatureSamplingStrategy(SamplingAssertions):
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

    def test_data_dimensions_correct(self):
        """Sampled data keeps every instance and ceil(features * fraction) columns."""
        strategy = FeatureSamplingStrategy(feature_fraction=0.3)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            # ceil(20 * 0.3) = 6
            self.assertEqual(result.data.shape, (100, 6))
            np.testing.assert_array_equal(result.labels, self.labels)

    def test_feature_mapping_matches_sampled_columns(self):
        """feature_mapping names exactly the parent columns present in the data."""
        strategy = FeatureSamplingStrategy(feature_fraction=0.3)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            self.assert_feature_mapping_well_formed(result, self.data.shape[1])
            self.assert_columns_match_mapping(result, self.data)

    def test_full_fraction_keeps_all_features_in_order(self):
        """With feature_fraction=1.0 children get the identity mapping."""
        strategy = FeatureSamplingStrategy(feature_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            np.testing.assert_array_equal(result.data, self.data)
            self.assertEqual(result.feature_mapping, {i: i for i in range(20)})

    def test_features_do_not_overlap_when_replace_false(self):
        """With replace=False no parent feature reaches two children."""
        strategy = FeatureSamplingStrategy(feature_fraction=0.3, replace=False)
        results = strategy.sample(self.data, self.labels, num_children=3)

        seen = set()
        for result in results:
            current = set(result.feature_mapping.values())
            self.assertEqual(
                len(current & seen), 0, f"features reused across children: {current}"
            )
            seen |= current

    def test_too_few_features_raises(self):
        """A fraction that yields fewer than MIN_FEATURES columns raises ValueError."""
        strategy = FeatureSamplingStrategy(feature_fraction=0.1)
        with self.assertRaises(ValueError):
            strategy.sample(self.data[:, :5], self.labels, num_children=2)

    def test_invalid_feature_fraction_raises(self):
        """feature_fraction <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            FeatureSamplingStrategy(feature_fraction=0.0)
        with self.assertRaises(ValueError):
            FeatureSamplingStrategy(feature_fraction=-1.0)


class TestInstanceSamplingStrategy(SamplingAssertions):
    """Tests for InstanceSamplingStrategy."""

    def setUp(self):
        np.random.seed(42)
        # Self-identifying rows, so tests can tell which instances a child received.
        self.data = identifiable_data(100, 20)
        self.labels = np.random.randint(0, 2, 100)

    def test_returns_correct_number_of_results(self):
        """sample() returns exactly num_children results."""
        strategy = InstanceSamplingStrategy(sample_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_children=5)
        self.assertEqual(len(results), 5)

    def test_data_dimensions_correct(self):
        """Sampled data keeps every feature and ceil(instances * fraction) rows."""
        strategy = InstanceSamplingStrategy(sample_fraction=0.3)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            # ceil(100 * 0.3) = 30
            self.assertEqual(result.data.shape, (30, 20))
            self.assertEqual(len(result.labels), 30)
            self.assertIsNone(result.feature_mapping)

    def test_rows_and_labels_are_sliced_together(self):
        """Each child label belongs to the instance sitting in the same row."""
        strategy = InstanceSamplingStrategy(sample_fraction=0.3)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            instances = recover_row_indices(result.data)
            np.testing.assert_array_equal(result.labels, self.labels[instances])

    def test_instances_are_unique_within_a_child(self):
        """No instance is handed to the same child twice."""
        strategy = InstanceSamplingStrategy(sample_fraction=0.3, replace=True)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            instances = recover_row_indices(result.data)
            self.assertEqual(len(np.unique(instances)), len(instances))

    def test_instances_do_not_overlap_when_replace_false(self):
        """With replace=False no instance reaches two children."""
        strategy = InstanceSamplingStrategy(sample_fraction=0.3, replace=False)
        results = strategy.sample(self.data, self.labels, num_children=3)

        seen = set()
        for result in results:
            current = set(recover_row_indices(result.data).tolist())
            self.assertEqual(
                len(current & seen), 0, f"instances reused across children: {current}"
            )
            seen |= current

    def test_full_fraction_keeps_all_instances(self):
        """With sample_fraction=1.0 children get the whole dataset."""
        strategy = InstanceSamplingStrategy(sample_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            np.testing.assert_array_equal(result.data, self.data)
            np.testing.assert_array_equal(result.labels, self.labels)

    def test_too_few_instances_raises(self):
        """A fraction that yields fewer than MIN_INSTANCES rows raises ValueError."""
        strategy = InstanceSamplingStrategy(sample_fraction=0.1)
        with self.assertRaises(ValueError):
            strategy.sample(self.data[:5], self.labels[:5], num_children=2)

    def test_invalid_sample_fraction_raises(self):
        """sample_fraction <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            InstanceSamplingStrategy(sample_fraction=0.0)
        with self.assertRaises(ValueError):
            InstanceSamplingStrategy(sample_fraction=-1.0)


class TestCombinedSamplingStrategy(SamplingAssertions):
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

    def test_data_dimensions_correct(self):
        """Both fractions are applied to the sampled data."""
        strategy = CombinedSamplingStrategy(feature_fraction=0.3, sample_fraction=0.3)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            # ceil(100 * 0.3) = 30 rows, ceil(20 * 0.3) = 6 columns
            self.assertEqual(result.data.shape, (30, 6))
            self.assertEqual(len(result.labels), 30)

    def test_feature_mapping_well_formed(self):
        """feature_mapping covers the child columns and names distinct parents."""
        strategy = CombinedSamplingStrategy(feature_fraction=0.3, sample_fraction=0.3)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            self.assert_feature_mapping_well_formed(result, self.data.shape[1])

    def test_rows_come_from_the_parent(self):
        """Sampled rows and labels are consistent with the parent's projection."""
        strategy = CombinedSamplingStrategy(feature_fraction=0.3, sample_fraction=0.3)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            self.assert_rows_exist_in_parent(result, self.data, self.labels)

    def test_columns_match_mapping_when_all_instances_kept(self):
        """With sample_fraction=1.0 the columns can be compared to the parent directly."""
        strategy = CombinedSamplingStrategy(feature_fraction=0.3, sample_fraction=1.0)
        results = strategy.sample(self.data, self.labels, num_children=3)

        for result in results:
            self.assert_columns_match_mapping(result, self.data)

    def test_features_do_not_overlap_when_replace_false(self):
        """With replace=False no parent feature reaches two children."""
        strategy = CombinedSamplingStrategy(
            feature_fraction=0.3, sample_fraction=0.3, replace=False
        )
        results = strategy.sample(self.data, self.labels, num_children=3)

        seen = set()
        for result in results:
            current = set(result.feature_mapping.values())
            self.assertEqual(
                len(current & seen), 0, f"features reused across children: {current}"
            )
            seen |= current

    def test_instances_do_not_overlap_when_replace_false(self):
        """With replace=False no instance reaches two children."""
        data = identifiable_data(100, 20)
        strategy = CombinedSamplingStrategy(
            feature_fraction=1.0, sample_fraction=0.3, replace=False
        )
        results = strategy.sample(data, self.labels, num_children=3)

        seen = set()
        for result in results:
            current = set(recover_row_indices(result.data).tolist())
            self.assertEqual(
                len(current & seen), 0, f"instances reused across children: {current}"
            )
            seen |= current

    def test_invalid_fractions_raise(self):
        """Invalid fractions raise ValueError."""
        with self.assertRaises(ValueError):
            CombinedSamplingStrategy(feature_fraction=0.0)
        with self.assertRaises(ValueError):
            CombinedSamplingStrategy(sample_fraction=-1.0)


class TestSamplingRandomized(SamplingAssertions):
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

    def test_feature_sampling_mapping_matches_sampled_columns(self):
        """FeatureSamplingStrategy keeps exactly the columns feature_mapping names."""

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
                self.assert_feature_mapping_well_formed(result, num_features)
                self.assert_columns_match_mapping(result, data)
                np.testing.assert_array_equal(result.labels, labels)

        self._run_randomized_test(check)

    def test_combined_sampling_mapping_and_rows_consistent(self):
        """CombinedSamplingStrategy rows and labels stay consistent with the parent."""

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
                self.assert_feature_mapping_well_formed(result, num_features)
                self.assert_rows_exist_in_parent(result, data, labels)

        self._run_randomized_test(check)


if __name__ == "__main__":
    unittest.main()
