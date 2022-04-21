"""
Testing Matrix methods
"""

from unittest import TestCase  # pragma: no cover

from ..matrices import Matrices  # pragma: no cover


class TestMatrices(TestCase):  # pragma: no cover
    """
    This class contains tests for Matrix methods
    """

    _MATRIX = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    _GENERATED_MATRIX = [[0, 1], [1, 0], [2, 1]]
    _IDENTITY_MATRIX = [[1, 0], [0, 1]]
    _DATA_FN = lambda self, i, j: i - j if i >= j else i + j

    def test_shape(self):
        """Successfully test"""
        self.assertEqual(Matrices.shape(matrix=self._MATRIX), (3, 3))

    def test_shape_bad_shape(self):
        """Unsuccessfully test"""
        self.assertIsNot(Matrices.shape(matrix=self._MATRIX), (3, 4))

    def test_get_row(self):
        """Successfully test"""
        self.assertEqual(Matrices.get_row(matrix=self._MATRIX, index=0), [1, 2, 3])

    def test_get_row_bad_row(self):
        """Unsuccessfully test"""
        self.assertIsNot(Matrices.get_row(matrix=self._MATRIX, index=1), [1, 2, 3])

    def test_get_column(self):
        """Successfully test"""
        self.assertEqual(Matrices.get_column(matrix=self._MATRIX, index=0), [1, 4, 7])

    def test_get_column_bad_column(self):
        """Unsuccessfully test"""
        self.assertIsNot(Matrices.get_column(matrix=self._MATRIX, index=1), [1, 4, 7])

    def test_make_matrix(self):
        """Successfully test"""
        generated_matrix = Matrices.make_matrix(num_rows=3, num_cols=2, entry_fn=self._DATA_FN)
        self.assertEqual(generated_matrix, self._GENERATED_MATRIX)

    def test_make_matrix_bad_matrix(self):
        """Unsuccessfully test"""
        generated_matrix = Matrices.make_matrix(num_rows=2, num_cols=3, entry_fn=self._DATA_FN)
        self.assertIsNot(generated_matrix, self._GENERATED_MATRIX)

    def test_identity_matrix(self):
        """Successfully test"""
        self.assertEqual(Matrices.identity_matrix(size=2), self._IDENTITY_MATRIX)

    def test_identity_matrix_bad_matrix(self):
        """Unsuccessfully test"""
        self.assertIsNot(Matrices.identity_matrix(size=3), self._IDENTITY_MATRIX)
