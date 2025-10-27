import doctest
import unittest

import hgp_lib


class TestMutations(unittest.TestCase):
    def test_doctests(self):
        result = doctest.testmod(hgp_lib.mutations.base_mutation, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.mutations.literal_mutations, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.mutations.operator_mutations, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")
        result = doctest.testmod(hgp_lib.mutations.mutation_executor, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
