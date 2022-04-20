"""
Vector methods implementation
"""

import math
from typing import List, Union, NoReturn


Vector = List[Union[float, int]]


class Vectors:
    """
    Linear algebra methods related to vectors to work just with numbers
    """

    @staticmethod
    def callable_validator(vectors: List[Vector]) -> NoReturn:
        """
        Validate if the input has the appropriate data type
        :param vectors: A list with vectors
        :rtype: NoReturn
        """
        if not all(isinstance(vector, list) for vector in vectors):
            raise TypeError("Input must be Vector type")

    @staticmethod
    def type_validator(vectors: List[Vector]) -> NoReturn:
        """
        Validate if the values inside the vectors are integers or floating points numbers
        :param vectors: A list with vectors
        :rtype: NoReturn
        """
        value_booleans = [[type(value) in [float, int] for value in vector] for vector in vectors]
        if not all((all(boolean) for boolean in value_booleans)):
            raise TypeError("Input must be float or integer values")

    @staticmethod
    def length_validator(vectors: List[Vector]) -> NoReturn:
        """
        Validate if two vectors have the same length
        :param vectors: A list with two vectors
        :rtype: NoReturn
        """
        length: int = len(vectors[0])
        if not all(len(vector) == length for vector in vectors):
            raise ValueError("Vectors should be the same length")

    @classmethod
    def add(cls, vector_x: Vector, vector_y: Vector) -> Vector:
        """
        Adds to vectors (vector_x + vector_y)
        :param vector_x: Vector with integers or floating point numbers
        :param vector_y: Vector with integers or floating point numbers
        :return: The sum of the vector_x and vector_y element-wise
        :rtype: Vector
        """
        cls.callable_validator(vectors=[vector_x, vector_y])
        cls.type_validator(vectors=[vector_x, vector_y])
        cls.length_validator(vectors=[vector_x, vector_y])
        return [x_i + y_i for x_i, y_i in zip(vector_x, vector_y)]

    @classmethod
    def subtract(cls, vector_x: Vector, vector_y: Vector) -> Vector:
        """
        Subtracts to vectors (vector_x - vector_y)
        :param vector_x: Vector with integers or floating point numbers
        :param vector_y: Vector with integers or floating point numbers
        :return: The subtraction of the vector_x and vector_y element-wise
        :rtype: Vector
        """
        cls.callable_validator(vectors=[vector_x, vector_y])
        cls.type_validator(vectors=[vector_x, vector_y])
        cls.length_validator(vectors=[vector_x, vector_y])
        return [x_i - y_i for x_i, y_i in zip(vector_x, vector_y)]

    @classmethod
    def vector_sum(cls, vectors: List[Vector]) -> Vector:
        """
        Sums all corresponding vectors. Like summing columns
        :param vectors: A list with two vectors
        :return: One vector with the sum of the i-th element of each vector input
        :rtype: Vector
        """
        if not vectors:
            raise TypeError("Not vectors provided.")
        cls.callable_validator(vectors=vectors)
        cls.type_validator(vectors=vectors)
        num_elements: int = len(vectors[0])
        if not all(len(v) == num_elements for v in vectors):
            raise ValueError("Vectors should be the same length")
        return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

    @classmethod
    def scalar_multiply(cls, scalar: Union[float, int], vector: Vector) -> Vector:
        """
        Multiplies every element by a scalar
        :param scalar: Integer or floating point number
        :param vector: Vector with integers or floating point numbers
        :return: The multiplication of each value inside the vector by the scalar
        :rtype: Vector
        """
        cls.callable_validator(vectors=[vector])
        cls.type_validator(vectors=[vector])
        if type(scalar) not in [float, int]:
            raise TypeError("Input 'scalar' should be an integer or a float")
        return [scalar * v_i for v_i in vector]

    @classmethod
    def vector_mean(cls, vectors: List[Vector]) -> Vector:
        """
        Computes the column-wise average
        :param vectors: A list of vectors
        :return: The average of each column
        :rtype: Vector
        """
        size: int = len(vectors)
        if size == 0:
            raise ZeroDivisionError("Vectors should contain information to do the operation")
        return cls.scalar_multiply(1/size, cls.vector_sum(vectors=vectors))

    @classmethod
    def dot(cls, vector_x: Vector, vector_y: Vector) -> [float or int]:
        """
        Computes dot product operation
        :param vector_x: Vector with integers or floating point numbers
        :param vector_y: Vector with integers or floating point numbers
        :return: The value of the dot product operation between two vectors
        :rtype: float or int
        """
        cls.length_validator([vector_x, vector_y])
        return sum(x_i * y_i for x_i, y_i in zip(vector_x, vector_y))

    @classmethod
    def sum_of_squares(cls, vector: Vector) -> [float or int]:
        """
        Computes the sum of the squares
        :param vector: Vector with integers or floating point numbers
        :return: The value of the sum of the squares operation for a vector
        :rtype: float or int
        """
        return cls.dot(vector_x=vector, vector_y=vector)

    @classmethod
    def magnitude(cls, vector: Vector) -> [float or int]:
        """
        Computes the magnitude (or length) of a vector
        :param vector: Vector with integers or floating point numbers
        :return: The value of the magnitude (or length) of a vector
        :rtype: float or int
        """
        return math.sqrt(cls.sum_of_squares(vector=vector))

    @classmethod
    def squared_distance(cls, vector_x: Vector, vector_y: Vector) -> [float or int]:
        """
        Computes (x_1 - y_1) ** 2 + ... + (x_n - y_n) ** 2
        :param vector_x: Vector with integers or floating point numbers
        :param vector_y: Vector with integers or floating point numbers
        :return: The value of the sum of the square of the difference between two vectors
        :rtype: float or int
        """
        return cls.sum_of_squares(vector=cls.subtract(vector_x=vector_x, vector_y=vector_y))

    @classmethod
    def distance(cls, vector_x: Vector, vector_y: Vector) -> [float or int]:
        """
        Computes the distance between vector_x and vector_y
        This is another option --> return magnitude(subtract(v, w))
        :param vector_x: Vector with integers or floating point numbers
        :param vector_y: Vector with integers or floating point numbers
        :return: The value of the distance between two vectors
        :rtype: float or int
        """
        return math.sqrt(cls.squared_distance(vector_x=vector_x, vector_y=vector_y))
