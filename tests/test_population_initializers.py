import unittest
import numpy as np

from hgp_lib.populations.initializers.random_initializer import RandomInitializer
from hgp_lib.rules import Literal
from hgp_lib.rules.operators import And, Or
from hgp_lib.rules.low_memory_operators import And as LowMemoryAnd, Or as LowMemoryOr


# TODO: Also provide a testing functionality that will test a client population initializer
class TestPopulationInitializers(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(10, 20) < 0.5
        self.labels = np.random.rand(10) < 0.5

    def test_random_initializer(self):
        initializer = RandomInitializer(100, self.data, self.labels, [And, Or], 25)
        self.assertEqual(initializer.max_num_literals, self.data.shape[1])

        with self.assertRaises(ValueError):  # We try to select more literals than available
            initializer.generate_one(self.data.shape[1] * 10, And, False)

        all_literals_and = initializer.generate_one(self.data.shape[1], And, False)
        columns = sorted([x.value for x in all_literals_and.subrules])

        self.assertIsInstance(all_literals_and, And)
        self.assertEqual(all_literals_and.negated, False)
        self.assertListEqual(columns, list(range(self.data.shape[1])))
        np.testing.assert_array_equal(all_literals_and.evaluate(self.data), self.data.all(axis=1))

        all_literals_or = initializer.generate_one(self.data.shape[1], Or, False)
        self.assertIsInstance(all_literals_or, Or)
        np.testing.assert_array_equal(all_literals_or.evaluate(self.data), self.data.any(axis=1))

        negative_operator = initializer.generate_one(10, Or, True)
        self.assertEqual(negative_operator.negated, True)

        multiple_operators = initializer.generate()
        for operator in multiple_operators:
            self.assertTrue(isinstance(operator, Or) or isinstance(operator, And))


if __name__ == "__main__":
    unittest.main()
