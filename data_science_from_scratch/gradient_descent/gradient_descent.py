"""
Gradient descent methods
"""

import random
from typing import Iterator, List, Tuple, TypeVar

from ..linear_algebra.vectors import Value, Vector, Vectors

T = TypeVar('T')


class GradientDescent:  # pylint: disable=R0913
    """
    Gradient descent methods implementation
    """

    @staticmethod
    def _gradient_step(vector: Vector, gradient: Vector,
                       step_size: float) -> Vector:  # pragma: no cover
        """
        Update gradient with the step
        :param vector: Vector
        :param gradient: Vector
        :param step_size: Size of the 'correction'
        :return: Vector
        :rtype: Vector
        """
        if len(vector) != len(gradient):
            raise ValueError("Gradient should have the same len than the vector")
        step = Vectors.scalar_multiply(scalar=step_size, vector=gradient)
        return Vectors.add(vector_x=vector, vector_y=step)

    @staticmethod
    def _sum_of_squares_gradient(vector: Vector) -> Vector:  # pragma: no cover
        """
        Sum of the squares of a gradient
        :param vector: Vector
        :return: Vector
        :rtype: Vector
        """
        return [2 * i for i in vector]

    @classmethod
    def reduce_distance(cls, steps: int, vector: Vector, step_size: float,
                        show: bool = True) -> Value:
        """
        Iteration to reduce the distance to [0, 0, 0] vector
        :param show: To show or not the progress
        :param step_size: Size of the 'correction'
        :param vector: Vector
        :param steps: Number of attempts
        :return: Value
        :rtype: Value
        """
        for epoch in range(steps):
            gradient = cls._sum_of_squares_gradient(vector=vector)
            vector = cls._gradient_step(vector=vector, gradient=gradient, step_size=step_size)
            if show:
                if epoch % (steps // 10) == 0 or epoch == (steps - 1):
                    print(epoch, vector)
        return Vectors.distance(vector_x=vector, vector_y=[0, 0, 0])

    @staticmethod
    def _linear_gradient(value_x: Value, value_y: Value,
                         theta: Vector) -> Vector:  # pragma: no cover
        """
        Gradient for a linear function
        :param value_x: Value
        :param value_y: Value
        :param theta: Vector
        :return: Vector
        :rtype: Vector
        """
        slope, intercept = theta
        predicted = (slope * value_x) + intercept
        error = predicted - value_y
        gradient = [2 * error * value_x, 2 * error]
        return gradient

    @classmethod
    def reduce_linear_gradient(cls, steps: int, learning_rate: float, inputs: List,
                               theta: Vector, batches: bool = True) -> Tuple:
        """
        Reduce gradient for a linear function
        :param batches: To use or not data set batches
        :param theta: Vector
        :param inputs: Vector
        :param steps: Number of attempts
        :param learning_rate: Size of the 'correction'
        :return: Tuple[Value, Value]
        :rtype: Tuple[Value, Value]
        """
        batch_size = len(inputs) // 10
        for epoch in range(steps):
            for batch in cls._minibatches(dataset=inputs, batch_size=batch_size):
                if batches:
                    gradient = Vectors.vector_mean(
                        [cls._linear_gradient(
                            value_x=value_x, value_y=value_y, theta=theta
                        ) for value_x, value_y in batch])
                    theta = cls._gradient_step(vector=theta, gradient=gradient,
                                               step_size=-learning_rate)
                else:
                    gradient = Vectors.vector_mean(
                        [cls._linear_gradient(
                            value_x=value_x, value_y=value_y, theta=theta
                        ) for value_x, value_y in inputs])
                    theta = cls._gradient_step(vector=theta, gradient=gradient,
                                               step_size=-learning_rate)
                    break
            if epoch % (steps // 10) == 0 or epoch == (steps - 1):
                print(epoch, theta)
        slope, intercept = theta
        return slope, intercept

    @staticmethod
    def _minibatches(dataset: List[T], batch_size: int,
                     shuffle: bool = True) -> Iterator[List[T]]:  # pragma: no cover
        """
        Batches creator
        :param dataset: Data set to split
        :param batch_size: Size of the batches to be generated
        :param shuffle: To shuffle the samples or not
        :return: Set of samples
        :rtype: Iterator[List[T]]
        """
        batch_starts = list(range(0, len(dataset), batch_size))
        if shuffle:
            random.shuffle(batch_starts)
        for start in batch_starts:
            end = start + batch_size
            yield dataset[start:end]
