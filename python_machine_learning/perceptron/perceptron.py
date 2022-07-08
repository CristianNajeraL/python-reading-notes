"""
Perceptron implementation
"""

import numpy as np


class Perceptron:  # pylint: disable=E1101
    """
    Perceptron classifier

    Attributes:
        weights_: 1d-array
            Weights after fitting
        errors_: list
            Number of mis classifications in each epoch
    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 17):
        """
        Class initializer
        :param eta: float
            Learning rate, between 0.0 and 1.0
        :param n_iter: int
            Passes over the training dataset
        :param random_state: int
            Random number generator seed for random weight initialization
        :return:
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.weights_: np.ndarray = np.ndarray([0, 0])
        self.errors_: list = []

    def net_input(self, x_values: np.ndarray) -> float:
        """
        Calculate net input
        :param x_values: nd-array
            Input values
        :return:
        """
        return np.dot(x_values, self.weights_[1:]) + self.weights_[0]

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        """
        Return class label after unit step
        :param x_values: nd-array
            Input values
        :return:
        """
        return np.where(self.net_input(x_values=x_values) >= 0.0, 1, -1)

    def fit(self, x_values: np.ndarray, y_values: np.ndarray) -> object:
        """
        Fit training data
        :param x_values: nd-array, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features
        :param y_values: 1d-array, shape = [n_examples]
            Target values
        :return:
        """
        np_random_state = np.random.RandomState(self.random_state)
        self.weights_: np.ndarray = np_random_state.normal(
            loc=0.0, scale=0.01, size=1 + x_values.shape[1])
        for _ in range(self.n_iter):
            errors: int = 0
            for x_value_i, target in zip(x_values, y_values):
                update = self.eta * (target - self.predict(x_value_i))
                self.weights_[1:] += update * x_value_i
                self.weights_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
