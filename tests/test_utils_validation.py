"""Tests for hgp_lib.utils.validation."""

import unittest

import numpy as np
import pandas as pd

from hgp_lib.rules import And, ComplexityCheck, Literal, Or
from hgp_lib.utils.validation import (
    BOOL_TYPES,
    check_isinstance,
    check_X,
    check_X_y,
    to_bool_labels,
    validate_callable,
    validate_num_literals,
    validate_operator_types,
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
        X = np.array([[True, False], [False, True]])
        labels = check_X_y(X, np.array([0, 1]))
        self.assertEqual(labels.dtype, bool)
        self.assertEqual(labels.tolist(), [False, True])

    def test_valid_dataframe(self):
        # A DataFrame holds raw features, so its dtypes are not checked.
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        check_X_y(X, np.array([0, 1]), x_type=pd.DataFrame)  # no error

    def test_int_data_is_rejected(self):
        with self.assertRaises(TypeError) as context:
            check_X_y(np.array([[1, 0], [0, 1]]), np.array([0, 1]))
        self.assertIn("BooleanRuleClassifier", str(context.exception))

    def test_float_labels_are_rejected(self):
        with self.assertRaises(TypeError):
            check_X_y(np.array([[True], [False]]), np.array([0.0, 1.0]))

    def test_labels_other_than_0_and_1_are_rejected(self):
        with self.assertRaises(ValueError):
            check_X_y(np.array([[True], [False]]), np.array([0, 2]))

    def test_bool_labels_are_returned_as_they_are(self):
        labels = np.array([True, False])
        self.assertIs(check_X_y(np.array([[True], [False]]), labels), labels)

    def test_x_none(self):
        with self.assertRaises(ValueError):
            check_X_y(None, np.array([1]))

    def test_y_none(self):
        with self.assertRaises(ValueError):
            check_X_y(np.array([[True]]), None)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            check_X_y(np.array([[True, False]]), np.array([0, 1]))

    def test_empty(self):
        with self.assertRaises(ValueError):
            check_X_y(np.empty((0, 2), dtype=bool), np.array([], dtype=bool))

    def test_x_not_2d(self):
        with self.assertRaises(ValueError):
            check_X_y(np.array([True, False, True]), np.array([0, 1, 0]))

    def test_y_not_1d(self):
        with self.assertRaises(ValueError):
            check_X_y(np.array([[True, False], [False, True]]), np.array([[0, 1]]))

    def test_wrong_x_type(self):
        with self.assertRaises(TypeError):
            check_X_y("not array", np.array([1]))

    def test_wrong_y_type(self):
        with self.assertRaises(TypeError):
            check_X_y(np.array([[True, False]]), [0])


class TestCheckX(unittest.TestCase):
    def test_bool_data(self):
        check_X(np.array([[True, False]]))  # no error

    def test_rejects_int_float_and_dataframes(self):
        for data in (
            np.array([[1, 0]]),
            np.array([[1.0, 0.0]]),
            pd.DataFrame({"a": [True]}),
        ):
            with self.assertRaises(TypeError):
                check_X(data)

    def test_rejects_other_shapes(self):
        with self.assertRaises(ValueError):
            check_X(np.array([True, False]))


class TestToBoolLabels(unittest.TestCase):
    def test_int_labels_are_converted(self):
        labels = to_bool_labels(np.array([1, 0, 1], dtype=np.int8))
        self.assertEqual(labels.dtype, bool)
        self.assertEqual(labels.tolist(), [True, False, True])

    def test_rejects_floats_strings_and_other_values(self):
        with self.assertRaises(TypeError):
            to_bool_labels(np.array([1.0, 0.0]))
        with self.assertRaises(TypeError):
            to_bool_labels(np.array(["a", "b"]))
        with self.assertRaises(ValueError):
            to_bool_labels(np.array([-1, 1]))


class TestCheckIsinstanceName(unittest.TestCase):
    def test_name_in_message(self):
        with self.assertRaises(TypeError) as context:
            check_isinstance("yes", BOOL_TYPES, "batched")
        self.assertIn("batched", str(context.exception))

    def test_numpy_bools_are_bools(self):
        check_isinstance(np.bool_(True), BOOL_TYPES, "flag")  # no error


if __name__ == "__main__":
    unittest.main()
