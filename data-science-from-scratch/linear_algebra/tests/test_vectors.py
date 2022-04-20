from unittest import TestCase
from ..vectors import Vectors


class TestVectors(TestCase):
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
        self.assertIsNone(Vectors._callable_validator(self._VECTORS))

    def test__callable_validator_type_error(self):
        with self.assertRaises(TypeError):
            Vectors._callable_validator(self._NOT_CALLABLE_VECTORS)

    def test__type_validator(self):
        self.assertIsNone(Vectors._type_validator(self._VECTORS))

    def test__type_validator_type_error(self):
        with self.assertRaises(TypeError):
            Vectors._type_validator(self._VECTORS_WITH_STRING)

    def test__length_validator(self):
        self.assertIsNone(Vectors._length_validator(vectors=self._VECTORS))

    def test__length_validator_no_same_length(self):
        with self.assertRaises(ValueError):
            Vectors._length_validator(vectors=self._VECTORS_NO_SAME_LENGTH)

    def test_add(self):
        self.assertEqual(Vectors.add(vector_x=self._VECTORS[0], vector_y=self._VECTORS[1]), [2, 4, 6])

    def test_subtract(self):
        self.assertEqual(Vectors.subtract(vector_x=self._VECTORS[0], vector_y=self._VECTORS[1]), [0, 0, 0])

    def test_vector_sum(self):
        self.assertEqual(Vectors.vector_sum(vectors=self._THREE_VECTORS), [3, 6, 9])

    def test_vector_sum_no_vector(self):
        with self.assertRaises(TypeError):
            Vectors.vector_sum(vectors=[])

    def test_vector_sum_no_same_length(self):
        with self.assertRaises(ValueError):
            Vectors.vector_sum(vectors=self._VECTORS_NO_SAME_LENGTH)

    def test_scalar_multiply(self):
        self.assertEqual(Vectors.scalar_multiply(scalar=2, vector=self._VECTORS[0]), [2, 4, 6])

    def test_scalar_multiply_scalar_wrong_type(self):
        with self.assertRaises(TypeError):
            Vectors.scalar_multiply(scalar='2', vector=self._VECTORS[0])

    def test_vector_mean(self):
        self.assertEqual(Vectors.vector_mean(vectors=self._VECTORS), [1, 2, 3])

    def test_vector_mean_zero_division_error(self):
        with self.assertRaises(ZeroDivisionError):
            Vectors.vector_mean(vectors=[])

    def test_dot(self):
        self.assertEqual(Vectors.dot(vector_x=self._VECTORS[0], vector_y=self._VECTORS[1]), 14)

    def test_sum_of_squares(self):
        self.assertEqual(Vectors.sum_of_squares(vector=self._VECTORS[0]), 14)

    def test_magnitude(self):
        self.assertEqual(Vectors.magnitude(vector=self._VECTORS[0]), 14 ** (1 / 2))

    def test_squared_distance(self):
        self.assertEqual(Vectors.squared_distance(vector_x=self._VECTORS[0], vector_y=self._VECTORS[1]), 0)

    def test_distance(self):
        self.assertEqual(Vectors.distance(vector_x=self._VECTORS[0], vector_y=self._VECTORS[1]), 0)
