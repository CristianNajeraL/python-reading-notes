"""
Simple Linear Regression Implementation
"""

import random
from typing import Tuple

import tqdm

from ..gradient_descent.gradient_descent import GradientDescent as gd
from ..linear_algebra.vectors import Value, Vector
from ..statistics.tools import Tools as t


class SimpleLinearRegression:
    """
    Simple Linear Regression methods
    """

    @staticmethod
    def predict(alpha: Value, beta: Value, value: Value) -> Value:
        """
        Takes an input and generates a prediction based on the calculated coefficients
        :param alpha: Constant
        :param beta: Coefficient
        :param value: Random value
        :return: Model prediction
        """
        return beta * value + alpha

    @classmethod
    def error(cls, alpha: Value, beta: Value, value: Value, actual: Value) -> Value:
        """
        Calculates the error between the prediction and the actual value
        :param alpha: Constant
        :param beta: Coefficient
        :param value: Random value input
        :param actual: Actual value output
        :return: Difference between the actual value and the prediction
        """
        return cls.predict(alpha=alpha, beta=beta, value=value) - actual

    @classmethod
    def sum_of_squared_errors(cls, alpha: Value, beta: Value, inputs: Vector,
                              outputs: Vector) -> Value:
        """
        Calculates the sum of the squared errors
        :param alpha: Constant
        :param beta: Coefficient
        :param inputs: Vector with input values
        :param outputs: Vector with actual values
        :return: Sum of the squared errors
        """
        return sum(cls.error(alpha=alpha, beta=beta, value=value_x, actual=value_y) ** 2
                   for value_x, value_y in zip(inputs, outputs))

    @staticmethod
    def least_squares_fit(inputs: Vector, outputs: Vector) -> Tuple[Value, Value]:
        """
        Given to vectors, find the least-squares values of alpha and beta
        :param inputs: Vector with input values
        :param outputs: Vector with actual values
        :return: Constant (alpha) and coefficient (beta) values
        """
        correlation = t.correlation(vector_x=inputs, vector_y=outputs)
        inputs_std = t.standard_deviation(vector=inputs)
        outputs_std = t.standard_deviation(vector=outputs)
        beta = correlation * outputs_std / inputs_std
        alpha = t.mean(vector=outputs) - beta * t.mean(vector=inputs)
        return alpha, beta

    @staticmethod
    def total_sum_of_squares(vector: Vector) -> Value:
        """
        The total squared variation of each value from the vector mean
        :param vector: Data input
        :return: Sum of the squares of each value in the vector
        """
        return sum(value ** 2 for value in t.deviation_mean(vector=vector))

    @classmethod
    def r_squared(cls, alpha: Value, beta: Value, inputs: Vector, outputs: Vector) -> Value:
        """
        The fraction of variation of the output captured by the model, which equals
        1 - the fraction of variation of the output not captured by the model
        :param alpha: Constant
        :param beta: Coefficient
        :param inputs: Vector with input values
        :param outputs: Vector with actual values
        :return: R squared value
        """
        sum_of_squared_errors = cls.sum_of_squared_errors(alpha=alpha, beta=beta,
                                                          inputs=inputs, outputs=outputs)
        total_sum_of_squares = cls.total_sum_of_squares(vector=outputs)
        return 1.0 - (sum_of_squared_errors / total_sum_of_squares)

    @classmethod
    def gradient_descent_fit(cls, inputs: Vector, outputs: Vector, epochs: int = 15000,
                             learning_rate: float = 0.00001) -> Vector:
        """
        Given to vectors, find the least-squares values of alpha and beta using gradient descent
        :param inputs: Vector with input values
        :param outputs: Vector with actual values
        :param epochs: Number of attempts
        :param learning_rate: Size of the 'correction'
        :return: Constant (alpha) and coefficient (beta) values
        """
        guess = [random.random(), random.random()]
        with tqdm.trange(epochs) as epoch:
            for _ in epoch:
                alpha, beta = guess
                gradient_alpha = sum(2 * cls.error(
                    alpha=alpha, beta=beta, value=value_x, actual=value_y)
                                     for value_x, value_y in zip(inputs, outputs))
                gradient_beta = sum(2 * cls.error(
                    alpha=alpha, beta=beta, value=value_x, actual=value_y) * value_x
                                    for value_x, value_y in zip(inputs, outputs))
                loss = cls.sum_of_squared_errors(alpha=alpha, beta=beta, inputs=inputs,
                                                 outputs=outputs)
                epoch.set_description(f"Loss: {loss:.3f}")
                guess = gd.gradient_step(vector=guess, gradient=[gradient_alpha, gradient_beta],
                                         step_size=-learning_rate)
        return guess
