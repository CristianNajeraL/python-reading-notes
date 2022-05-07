"""
Multiple Regression implementation module
"""

import random
import tqdm

from typing import List

from ..gradient_descent.gradient_descent import GradientDescent as gd
from ..linear_algebra.vectors import Value, Vector
from ..linear_algebra.vectors import Vectors as v
from ..simple_linear_regression import SimpleLinearRegression as slr


class MultipleRegression:
    """
    Multiple regression implementation class
    """

    @staticmethod
    def predict(values: Vector, betas: Vector) -> Value:
        """
        Takes an input and generates a prediction based on the calculated coefficients
        :param betas: Coefficients to make the prediction
        :param values: Input values
        :return: Model prediction
        """
        return v.dot(vector_x=values, vector_y=betas)

    @classmethod
    def error(cls, values: Vector, value_y: Value, betas: Vector) -> Value:
        """
        Calculates the error between the prediction and the actual value
        :param betas: Coefficients to make the prediction
        :param values: Input values
        :param value_y: Actual value
        :return: Difference between the prediction and the actual value
        """
        return cls.predict(values=values, betas=betas) - value_y

    @classmethod
    def squared_error(cls, values: Vector, value_y: Value, betas: Vector) -> Value:
        """
        Returns the squared error of a prediction
        :param values: Input values
        :param value_y: Actual value
        :param betas: Coefficients to make the prediction
        :return: Squared error of a prediction
        """
        return cls.error(values=values, betas=betas, value_y=value_y) ** 2

    @classmethod
    def squared_error_gradient(cls, values: Vector, value_y: Value, betas: Vector) -> Vector:
        """
        Computes the gradient of the function
        :param values: Input values
        :param value_y: Actual value
        :param betas: Coefficients to make the prediction
        :return: Gradient of the function
        """
        error = cls.error(values=values, value_y=value_y, betas=betas)
        return [2 * error * value for value in values]

    @classmethod
    def least_squares_fit(cls, values_x: List[Vector], values_y: Vector,
                          learning_rate: float = 0.001, num_steps: int = 1000,
                          batch_size: int = 1) -> Vector:
        """
        Find the beta that minimizes the sum of squared errors
            assuming the model y = dot(x, beta)
        :param values_x: Input values
        :param values_y: Output values
        :param learning_rate: Size of the 'correction'
        :param num_steps: Number of attempts
        :param batch_size: Number of samples per batch
        :return: Optimal betas
        """
        guess = [random.random() for _ in values_x[0]]
        for _ in tqdm.trange(num_steps, desc="Least Squares Fit"):
            for start in range(0, len(values_x), batch_size):
                batch_x = values_x[start:start + batch_size]
                batch_y = values_y[start:start + batch_size]
                gradient = v.vector_mean(vectors=[cls.squared_error_gradient(
                    values=x, value_y=y, betas=guess) for x, y in zip(batch_x, batch_y)])
                guess = gd.gradient_step(vector=guess, gradient=gradient, step_size=-learning_rate)
        return guess

    @classmethod
    def multiple_r_squared(cls, values_x: List[Vector], values_y: Vector, betas: Vector) -> Value:
        """
        The fraction of variation of the output captured by the model, which equals
            1 - the fraction of variation of the output not captured by the model
        :param values_x: Input values
        :param values_y: Output values
        :param betas: Coefficients to make the prediction
        """
        sum_of_squared_errors = sum(
            cls.error(
                values=value_x,
                value_y=value_y,
                betas=betas
            ) ** 2 for value_x, value_y in zip(values_x, values_y))
        return 1.0 - sum_of_squared_errors / slr.total_sum_of_squares(values_y)
