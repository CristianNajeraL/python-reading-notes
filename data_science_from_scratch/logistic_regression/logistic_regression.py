"""
Logistic Regression implementation module
"""

import math
from typing import List, Tuple

import tqdm

from ..gradient_descent.gradient_descent import GradientDescent as Gd
from ..linear_algebra.vectors import Value, Vector, Vectors


class LogisticRegression:  # pylint: disable=R0913
    """
    Logistic regression implementation class
    """

    @staticmethod
    def logistic(value_x: Value) -> Value:
        """
        Execute logistic function
        :param value_x: Value to be used
        :return: Value after logistic function
        """
        return 1.0 / (1 + math.exp(-value_x))

    @classmethod
    def logistic_prime(cls, value_x: float) -> float:
        """
        Execute logistic function derivative
        :param value_x: Value to be used
        :return: Value after derivative on logistic function
        """
        prediction = cls.logistic(value_x=value_x)
        return prediction * (1 - prediction)

    @classmethod
    def _negative_log_likelihood(cls, value_x: Vector, value_y: Value, beta: Vector) -> Value:
        """
        The negative likelihood for one data point
        :param value_x: Input values
        :param value_y: Output value
        :param beta: Coefficients vector
        :return: Negative likelihood
        """
        if value_y == 1:
            return -math.log(cls.logistic(value_x=Vectors.dot(vector_x=value_x, vector_y=beta)))
        return -math.log(1 - cls.logistic(value_x=Vectors.dot(vector_x=value_x, vector_y=beta)))

    @classmethod
    def negative_log_likelihood(cls, value_x: List[Vector], value_y: List[Value],
                                beta: Vector) -> Value:
        """
        The negative likelihood of an array of data points
        :param value_x: Input values
        :param value_y: Output value
        :param beta: Coefficients vector
        :return: Negative likelihood
        """
        return sum(
            cls._negative_log_likelihood(
                value_x=x, value_y=y, beta=beta
            ) for x, y in zip(value_x, value_y)
        )

    @classmethod
    def _negative_log_partial_j(cls, value_x: Vector, value_y: Value, beta: Vector,
                                value_j: Value) -> Value:
        """
        The jth partial derivative for one data point
        :param value_x: Input values
        :param value_y: Output value
        :param beta: Coefficients vector
        :param value_j: Index of the data point
        :return: Partial derivative
        """
        return -(
                value_y - cls.logistic(value_x=Vectors.dot(vector_x=value_x, vector_y=beta))
        ) * value_x[value_j]

    @classmethod
    def _negative_log_gradient(cls, value_x: Vector, value_y: Value, beta: Vector) -> Vector:
        """
        The gradient for one data point
        :param value_x: Input values
        :param value_y: Output value
        :param beta: Coefficients vector
        :return: Gradient value
        """
        return [cls._negative_log_partial_j(value_x=value_x, value_y=value_y, beta=beta,
                                            value_j=j) for j in range(len(beta))]

    @classmethod
    def negative_log_gradient(cls, value_x: List[Vector], value_y: List[float],
                              beta: Vector) -> Vector:
        """
        The gradient for an array of data points
        :param value_x: Input values
        :param value_y: Output value
        :param beta: Coefficients vector
        :return: Gradient value
        """
        return Vectors.vector_sum(
            vectors=[
                cls._negative_log_gradient(
                    value_x=x, value_y=y, beta=beta) for x, y in zip(value_x, value_y)
            ]
        )

    @classmethod
    def fit_model(cls, beta: Vector, x_train: List[Vector], y_train: Vector, steps: Value,
                  learning_rate: Value = 0.01) -> Tuple:
        """
        Fit model with available data
        :param learning_rate: Correction step size
        :param steps: Number of loops to execute
        :param beta: Coefficients vector
        :param x_train: Input values
        :param y_train: Output values
        :return: Beta vector and loss value
        """
        with tqdm.trange(steps) as step:
            for _ in step:
                gradient = cls.negative_log_gradient(value_x=x_train, value_y=y_train, beta=beta)
                beta = Gd.gradient_step(vector=beta, gradient=gradient, step_size=-learning_rate)
                loss = cls.negative_log_likelihood(value_x=x_train, value_y=y_train, beta=beta)
                step.set_description(f"loss: {loss:.3f} beta: {beta}")
        return beta, loss

    @classmethod
    def precision_recall(cls, x_test: List[Vector], y_test: Vector, beta: Vector,
                         threshold: Value = 0.5) -> Tuple:
        """
        Evaluate precision and recall of the model
        :param threshold: Probability to be considered True
        :param beta: Coefficients vector
        :param x_test: Input values
        :param y_test: Output values
        :return: Precision and recall metrics
        """
        true_positives = false_positives = true_negatives = false_negatives = 0
        for value_x, value_y in zip(x_test, y_test):
            prediction = cls.logistic(Vectors.dot(vector_x=beta, vector_y=value_x))
            if value_y == 1 and prediction >= threshold:
                true_positives += 1
            elif value_y == 1:
                false_negatives += 1
            elif prediction >= threshold:
                false_positives += 1
            else:
                true_negatives += 1
        precision: Value = true_positives / (true_positives + false_positives)
        recall: Value = true_positives / (true_positives + false_negatives)
        return precision, recall
