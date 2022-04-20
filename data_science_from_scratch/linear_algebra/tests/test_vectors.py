"""
Testing vectors methods
"""

from unittest import TestCase
from ..vectors import Vectors


class TestVectors(TestCase):
    """
    This class contains tests for vector methods
    """
    _VECTORS = [
        [1, 2, 3],
        [1, 2, 3]
    ]
    _VECTORS_NO_SAME_LENGTH = [
        [1, 2, 3],
        [1, 2]
    ]
    _THREE_VECTORS = [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]
    ]
    _VECTORS_WITH_STRING = [
        [1, 2, 3],
        ['1', '2', '3']
    ]
    _NOT_CALLABLE_VECTORS = [1, 2]

    def test__callable_validator(self):
        """Successfully test of callable validator function"""
        self.assertIsNone(Vectors.callable_validator(self._VECTORS))

    def test__callable_validator_type_error(self):
        """Unsuccessfully test of callable validator function"""
        with self.assertRaises(TypeError):
            Vectors.callable_validator(self._NOT_CALLABLE_VECTORS)

    def test__type_validator(self):
        """Successfully test of type validator function"""
        self.assertIsNone(Vectors.type_validator(self._VECTORS))

    def test__type_validator_type_error(self):
        """Unsuccessfully test of type validator function"""
        with self.assertRaises(TypeError):
            Vectors.type_validator(self._VECTORS_WITH_STRING)

    def test__length_validator(self):
        """Successfully test of length validator function"""
        self.assertIsNone(Vectors.length_validator(vectors=self._VECTORS))

    def test__length_validator_no_same_length(self):
        """Unsuccessfully test of length validator function"""
        with self.assertRaises(ValueError):
            Vectors.length_validator(vectors=self._VECTORS_NO_SAME_LENGTH)

    def test_add(self):
        """Successfully test of add function"""
        add_return = Vectors.add(vector_x=self._VECTORS[0], vector_y=self._VECTORS[1])
        self.assertEqual(add_return, [2, 4, 6])

    def test_subtract(self):
        """Successfully test of subtract function"""
        subtract_return = Vectors.subtract(vector_x=self._VECTORS[0], vector_y=self._VECTORS[1])
        self.assertEqual(subtract_return, [0, 0, 0])

    def test_vector_sum(self):
        """Successfully test of vector sum function"""
        self.assertEqual(Vectors.vector_sum(vectors=self._THREE_VECTORS), [3, 6, 9])

    def test_vector_sum_no_vector(self):
        """Unsuccessfully test of vector sum function"""
        with self.assertRaises(TypeError):
            Vectors.vector_sum(vectors=[])

    def test_vector_sum_no_same_length(self):
        """Unsuccessfully test of vector sum function"""
        with self.assertRaises(ValueError):
            Vectors.vector_sum(vectors=self._VECTORS_NO_SAME_LENGTH)

    def test_scalar_multiply(self):
        """Successfully test of scalar multiply function"""
        self.assertEqual(Vectors.scalar_multiply(scalar=2, vector=self._VECTORS[0]), [2, 4, 6])

    def test_scalar_multiply_scalar_wrong_type(self):
        """Unsuccessfully test of scalar multiply function"""
        with self.assertRaises(TypeError):
            Vectors.scalar_multiply(scalar='2', vector=self._VECTORS[0])

    def test_vector_mean(self):
        """Successfully test of vector mean function"""
        self.assertEqual(Vectors.vector_mean(vectors=self._VECTORS), [1, 2, 3])

    def test_vector_mean_zero_division_error(self):
        """Unsuccessfully test of distance function"""
        with self.assertRaises(ZeroDivisionError):
            Vectors.vector_mean(vectors=[])

    def test_dot(self):
        """Successfully test of dot function"""
        self.assertEqual(Vectors.dot(vector_x=self._VECTORS[0], vector_y=self._VECTORS[1]), 14)

    def test_sum_of_squares(self):
        """Successfully test of sum of squares function"""
        self.assertEqual(Vectors.sum_of_squares(vector=self._VECTORS[0]), 14)

    def test_magnitude(self):
        """Successfully test of magnitude function"""
        self.assertEqual(Vectors.magnitude(vector=self._VECTORS[0]), 14 ** (1 / 2))

    def test_squared_distance(self):
        """Successfully test of squared distance function"""
        s_d_return = Vectors.squared_distance(vector_x=self._VECTORS[0], vector_y=self._VECTORS[1])
        self.assertEqual(s_d_return, 0)

    def test_distance(self):
        """Successfully test of distance function"""
        self.assertEqual(Vectors.distance(vector_x=self._VECTORS[0], vector_y=self._VECTORS[1]), 0)
