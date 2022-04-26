"""
Gradient descent methods
"""


from typing import Callable

from ..linear_algebra.vectors import Value, Vector, Vectors


class GradientDescent:
    """
    Gradient descent methods implementation
    """

    @staticmethod
    def sum_of_squares(vector: Vector) -> Value:
        """

        :param vector:
        :return:
        """
        return Vectors.dot(vector_x=vector, vector_y=vector)

    @staticmethod
    def difference_quotient(function: Callable[[float], float], value: Value,
                            interval_length: Value) -> Value:
        """

        :param function:
        :param value:
        :param interval_length:
        :return:
        """
        return (function(value + interval_length) - function(value)) / interval_length

    @staticmethod
    def square(value: Value) -> Value:
        """

        :param value:
        :return:
        """
        return value * value

    @staticmethod
    def derivate(value: Value) -> Value:
        """

        :param value:
        :return:
        """
        return 2 * value

    @staticmethod
    def partial_difference_quotient(function: Callable[[Vector], float], vector: Vector,
                                    value: Value, interval_length: Value) -> Value:
        """

        :param function:
        :param vector:
        :param value:
        :param interval_length:
        :return:
        """
        weights = [v_j + (interval_length if j == value else 0) for j, v_j in enumerate(vector)]
        return (function(weights) - function(vector)) / interval_length

