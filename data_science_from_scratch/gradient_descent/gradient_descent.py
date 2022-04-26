"""
Gradient descent methods
"""


import random
from typing import Callable, Iterator, List, Tuple, TypeVar

from ..linear_algebra.vectors import Value, Vector, Vectors


T = TypeVar('T')


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

    @classmethod
    def estimate_gradient(cls, function: Callable[[Vector], float], vector: Vector,
                          interval_length: Value = 0.0001) -> Vector:
        """

        :param function:
        :param vector:
        :param interval_length:
        :return:
        :rtype: Vector
        """
        return [cls.partial_difference_quotient(function=function, vector=vector,
                                                value=value, interval_length=interval_length)
                for value in range(len(vector))]

    @staticmethod
    def gradient_step(vector: Vector, gradient: Vector, step_size: float) -> Vector:
        """

        :param vector:
        :param gradient:
        :param step_size:
        :return:
        """
        if len(vector) != len(gradient):
            raise ValueError("Gradient should have the same len than the vector")
        step = Vectors.scalar_multiply(scalar=step_size, vector=vector)
        return Vectors.add(vector_x=vector, vector_y=step)

    @staticmethod
    def sum_of_squares_gradient(vector: Vector) -> Vector:
        """

        :param vector:
        :return:
        """
        return [2 * i for i in vector]

    @classmethod
    def reduce_distance(cls, steps: int, vector: Vector, step_size: float, show: bool = True) -> Value:
        """

        :param show:
        :param step_size:
        :param vector:
        :param steps:
        :return:
        """
        for epoch in range(steps):
            gradient = cls.sum_of_squares_gradient(vector=vector)
            vector = cls.gradient_step(vector=vector, gradient=gradient, step_size=step_size)
            if show:
                if epoch % steps // 10 == 0 or epoch == steps - 1:
                    print(epoch, vector)
        return Vectors.distance(vector_x=vector, vector_y=[0, 0, 0])

    @staticmethod
    def linear_gradient(value_x: Value, value_y: Value, theta: Vector) -> Vector:
        """

        :param value_x:
        :param value_y:
        :param theta:
        :return:
        """
        slope, intercept = theta
        predicted = slope * value_x + intercept
        error = (predicted - value_y)
        gradient = [2 * error * value_x, 2 * error]
        return gradient

    @classmethod
    def reduce_linear_gradient(cls, steps: int, learning_rate: float, inputs: Vector,
                               theta: Vector, show: bool = True, batches: bool = True) -> Tuple:
        """

        :param batches:
        :param show:
        :param theta:
        :param inputs:
        :param steps:
        :param learning_rate:
        :return:
        """
        for epoch in range(steps):
            for batch in cls.minibatches(dataset=inputs, batch_size=20):
                if batches:
                    gradient = Vectors.vector_mean(
                        [cls.linear_gradient(
                            value_x=value_x, value_y=value_y, theta=theta) for value_x, value_y in batch])
                    theta = cls.gradient_step(vector=theta, gradient=gradient, step_size=-learning_rate)
                else:
                    gradient = Vectors.vector_mean(
                        [cls.linear_gradient(
                            value_x=value_x, value_y=value_y, theta=theta) for value_x, value_y in inputs])
                    theta = cls.gradient_step(vector=theta, gradient=gradient, step_size=-learning_rate)
                    break
            if show:
                if epoch % steps // 10 == 0 or epoch == steps - 1:
                    print(epoch, theta)
        slope, intercept = theta
        return slope, intercept

    @staticmethod
    def minibatches(dataset: List[T], batch_size: int, shuffle: bool = True) -> Iterator[List[T]]:
        """

        :param dataset:
        :param batch_size:
        :param shuffle:
        :return:
        """
        batch_starts = [start for start in range(0, len(dataset), batch_size)]
        if shuffle:
            random.shuffle(batch_starts)
        for start in batch_starts:
            end = start + batch_size
            yield dataset[start:end]
