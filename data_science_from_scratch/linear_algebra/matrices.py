"""
Linear algebra methods related to matrices
"""


from typing import Callable, List, Tuple

from .vectors import Vector


Matrix = List[List[float]]


class Matrices:
    """
    Matrices methods implementation
    """

    @staticmethod
    def shape(matrix: Matrix) -> Tuple[int, int]:
        """
        Return how many rows and columns a matrix has
        :param matrix: Matrix
        :return: A tuple with the numbers of rows, columns
        :rtype: Tuple[int, int]
        """
        num_rows = len(matrix)
        num_cols = len(matrix[0]) if matrix else 0
        return num_rows, num_cols

    @staticmethod
    def get_row(matrix: Matrix, index: int) -> Vector:
        """
        Return the i-th row of matrix (as a Vector)
        :param matrix: Matrix
        :param index: Integer number
        :return: A Vector representing the row
        :rtype: Vector
        """
        return matrix[index]

    @staticmethod
    def get_column(matrix: Matrix, index: int) -> Vector:
        """
        Return the j-th column of matrix (as a Vector)
        :param matrix: Matrix
        :param index: Integer number
        :return: A Vector representing the column
        :rtype: Vector
        """
        return [row[index] for row in matrix]

    @staticmethod
    def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
        """
        Return a Matrix with a num_rows, num_cols shape whose (i, j)-th entry is entry_fin(i, j)
        :param num_rows: Integer
        :param num_cols: Integer
        :param entry_fn: Callable function that is going to generate the values for the Matrix
        :return: Matrix with num_rows, num_cols shape
        :rtype: Matrix
        """
        return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]

    @classmethod
    def identity_matrix(cls, size: int) -> Matrix:
        """
        Returns an identity Matrix with size, size shape
        :param size: Integer
        :return: Matrix with size, size shape
        :rtype: Matrix
        """
        data_fn = lambda i, j: 1 if i == j else 0
        return cls.make_matrix(num_rows=size, num_cols=size, entry_fn=data_fn)
