"""
Multiple Regression implementation module
"""

import random
from typing import Callable, List, Tuple, TypeVar, Union

import tqdm

from ..gradient_descent.gradient_descent import GradientDescent as gd
from ..linear_algebra.vectors import Value, Vector
from ..linear_algebra.vectors import Vectors as v
from ..probability.probability import Probability as prob
from ..simple_linear_regression import SimpleLinearRegression as slr

X = TypeVar('X')
Stat = TypeVar('Stat')


class MultipleRegression:  # pylint: disable=R0913
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
    def least_squares_fit(cls, values_x: List[Vector], values_y: Union[Vector, List[Vector]],
                          learning_rate: float = 1e-3, num_steps: int = 1000,
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
        :return: R squared value
        """
        sum_of_squared_errors = sum(
            cls.error(
                values=value_x,
                value_y=value_y,
                betas=betas
            ) ** 2 for value_x, value_y in zip(values_x, values_y))
        return 1.0 - sum_of_squared_errors / slr.total_sum_of_squares(values_y)

    @staticmethod
    def _bootstrap_sample(data: List[X]) -> List[X]:
        """
        Randomly samples len(data) elements with replacement
        :param data: List with elements
        :return: Sample from data with replacement
        """
        return [random.choice(data) for _ in data]

    @classmethod
    def bootstrap_statistic(cls, data: List[X], stats_fn: Callable[[List[X]], Stat],
                            num_samples: int) -> List[Stat]:
        """
        Evaluates stats_fn on num_samples bootstrap samples from data
        """
        statistic = [stats_fn(cls._bootstrap_sample(data)) for _ in range(num_samples)]
        print(statistic)
        return statistic

    @classmethod
    def estimate_sample_beta(cls, pairs: List[Tuple[Vector, float]],
                             num_steps: int = 100, batch_size: int = 50) -> Vector:
        """
        Fits the least squares with x's and y's tuples
        """
        x_sample = [x for x, _ in pairs]
        y_sample = [y for _, y in pairs]
        beta = cls.least_squares_fit(values_x=x_sample, values_y=y_sample,
                                     num_steps=num_steps, batch_size=batch_size)
        return beta

    @staticmethod
    def p_value(beta_hat_j: float, sigma_hat_j: float) -> float:
        """
        Calculates p-value
        If the coefficient is positive, we need to compute twice the
            probability of seeing an even larger value, otherwise
            twice the probability of seeing a smaller value
        """
        if beta_hat_j > 0:
            return 2 * (1 - prob.normal_cdf(beta_hat_j / sigma_hat_j))
        return 2 * prob.normal_cdf(beta_hat_j / sigma_hat_j)

    @staticmethod
    def _ridge_penalty(betas: Vector, alpha: float) -> Value:
        """
        Estimate ridge penalty
        """
        return alpha * v.dot(vector_x=betas[1:], vector_y=betas[1:])

    @classmethod
    def squared_error_ridge(cls, values_x: Vector, value_y: float, betas: Vector,
                            alpha: float) -> Value:
        """
        Estimate error plus ridge penalty on betas
        """
        return cls.error(values=values_x, value_y=value_y, betas=betas) ** 2 + cls._ridge_penalty(
            betas=betas, alpha=alpha)

    @staticmethod
    def _ridge_penalty_gradient(betas: Vector, alpha: Value) -> Vector:
        """
        Gradient of just the ridge penalty
        """
        return [0.] + [2 * alpha * beta_j for beta_j in betas[1:]]

    @classmethod
    def squared_error_ridge_penalty(cls, values_x: Vector, value_y: Value, betas: Vector,
                                    alpha: Value) -> Vector:
        """
        The gradient corresponding to the ith squared error term including the ridge penalty
        """
        return v.add(vector_x=cls.squared_error_gradient(
            values=values_x, value_y=value_y, betas=betas),
            vector_y=cls._ridge_penalty_gradient(
                betas=betas, alpha=alpha))

    @classmethod
    def least_squares_fit_ridge(cls, values_x: List[Vector], values_y: Union[Vector, List[Vector]],
                                alpha: Value = 0.0, learning_rate: float = 1e-3,
                                num_steps: int = 1000,
                                batch_size: int = 1) -> Vector:
        """
        Find the beta that minimizes the sum of squared errors
            assuming the model y = dot(x, beta)
        :param values_x: Input values
        :param values_y: Output values
        :param alpha: Ridge penalty rate
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
                gradient = v.vector_mean(vectors=[
                    cls.squared_error_ridge_penalty(
                        values_x=x,
                        value_y=y,
                        betas=guess,
                        alpha=alpha) for x, y in zip(batch_x, batch_y)])
                guess = gd.gradient_step(
                    vector=guess, gradient=gradient, step_size=-learning_rate)
        return guess
