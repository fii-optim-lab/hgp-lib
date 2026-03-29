"""Tests for hgp_lib.utils.validation."""

import doctest
import unittest

import numpy as np
import pandas as pd

import hgp_lib.utils.validation
from hgp_lib.rules import Literal, And, Or
from hgp_lib.utils.validation import (
    ComplexityCheck,
    validate_callable,
    check_isinstance,
    validate_num_literals,
    validate_operator_types,
    check_X_y,
)


class TestCheckComplexity(unittest.TestCase):
    def test_within_limit(self):
        self.assertTrue(ComplexityCheck(5)(Literal(value=0)))

    def test_at_limit(self):
        rule = And([Literal(value=0), Literal(value=1)])  # 3 nodes
        self.assertTrue(ComplexityCheck(3)(rule))

    def test_exceeds_limit(self):
        rule = And([Literal(value=0), Literal(value=1)])  # 3 nodes
        self.assertFalse(ComplexityCheck(2)(rule))


class TestComplexityCheck(unittest.TestCase):
    def test_returns_callable(self):
        check = ComplexityCheck(10)
        self.assertTrue(callable(check))

    def test_accepts_small_rule(self):
        check = ComplexityCheck(5)
        self.assertTrue(check(Literal(value=0)))

    def test_rejects_large_rule(self):
        check = ComplexityCheck(2)
        rule = And([Literal(value=0), Literal(value=1)])  # 3 nodes
        self.assertFalse(check(rule))

    def test_boundary(self):
        check = ComplexityCheck(3)
        rule = And([Literal(value=0), Literal(value=1)])  # exactly 3
        self.assertTrue(check(rule))


class TestValidateCallable(unittest.TestCase):
    def test_callable_passes(self):
        validate_callable(len)  # no error

    def test_non_callable_raises(self):
        with self.assertRaises(TypeError):
            validate_callable(42)

    def test_custom_error_message(self):
        with self.assertRaises(TypeError) as ctx:
            validate_callable("bad", error_message="custom msg")
        self.assertIn("custom msg", str(ctx.exception))

    def test_default_error_message(self):
        with self.assertRaises(TypeError) as ctx:
            validate_callable(123)
        self.assertIn("score_fn must be callable", str(ctx.exception))


class TestCheckIsinstance(unittest.TestCase):
    def test_correct_type_passes(self):
        check_isinstance(42, int)  # no error

    def test_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            check_isinstance("hello", int)

    def test_tuple_of_types(self):
        check_isinstance(42, (int, float))  # no error

    def test_tuple_of_types_wrong(self):
        with self.assertRaises(TypeError):
            check_isinstance("hello", (int, float))


class TestValidateNumLiterals(unittest.TestCase):
    def test_valid(self):
        validate_num_literals(5)  # no error

    def test_boundary(self):
        validate_num_literals(2)  # no error

    def test_too_small(self):
        with self.assertRaises(ValueError):
            validate_num_literals(1)

    def test_zero(self):
        with self.assertRaises(ValueError):
            validate_num_literals(0)

    def test_not_int(self):
        with self.assertRaises(TypeError):
            validate_num_literals(2.5)


class TestValidateOperatorTypes(unittest.TestCase):
    def test_valid(self):
        validate_operator_types((And, Or))  # no error

    def test_too_few(self):
        with self.assertRaises(ValueError):
            validate_operator_types((And,))

    def test_non_rule_type(self):
        with self.assertRaises(TypeError):
            validate_operator_types((And, str))

    def test_not_sequence(self):
        with self.assertRaises(TypeError):
            validate_operator_types(And)


class TestCheckXy(unittest.TestCase):
    def test_valid_numpy(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        check_X_y(X, y)  # no error

    def test_valid_dataframe(self):
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = np.array([0, 1])
        check_X_y(X, y, x_type=pd.DataFrame)  # no error

    def test_x_none(self):
        with self.assertRaises(ValueError):
            check_X_y(None, np.array([1]))

    def test_y_none(self):
        with self.assertRaises(ValueError):
            check_X_y(np.array([[1]]), None)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            check_X_y(np.array([[1, 2]]), np.array([0, 1]))

    def test_empty(self):
        with self.assertRaises(ValueError):
            check_X_y(np.empty((0, 2)), np.array([]))

    def test_x_not_2d(self):
        with self.assertRaises(ValueError):
            check_X_y(np.array([1, 2, 3]), np.array([0, 1, 0]))

    def test_y_not_1d(self):
        with self.assertRaises(ValueError):
            check_X_y(np.array([[1, 2], [3, 4]]), np.array([[0, 1]]))

    def test_wrong_x_type(self):
        with self.assertRaises(TypeError):
            check_X_y("not array", np.array([1]))

    def test_wrong_y_type(self):
        with self.assertRaises(TypeError):
            check_X_y(np.array([[1, 2]]), [0])


class TestDoctests(unittest.TestCase):
    def test_doctests(self):
        result = doctest.testmod(hgp_lib.utils.validation, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
