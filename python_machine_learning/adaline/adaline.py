"""
Adaline classifier implementation
"""

from typing import NoReturn

import numpy as np


class Adaline:  # pylint: disable=E1101 disable=R0902
    """
    Adaptive Linear Neuron classifier with Gradient Descendent

    Attributes:
        weights_: 1d-array
            Weights after fitting
        cost_: list
            Sum-of-squares cost function value in each epoch
    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50,
                 random_state: int = 17, shuffle: bool = True):
        """
        Class initializer
        :param eta: float
            Learning rate, between 0.0 and 1.0
        :param n_iter: int
            Passes over the training dataset
        :param random_state: int
            Random number generator seed for random weight initialization
        :param shuffle: bool
            Shuffles training data every epoch if True to prevent cycles
        :return:
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.weights_: np.ndarray = np.ndarray([0, 0])
        self.cost_: list = []
        self.shuffle: bool = shuffle
        self.np_random_state: object = object
        self.weights_initialized: bool = False

    @staticmethod
    def activation(x_values: float) -> float:
        """
        Compute linear activation
        :param x_values: nd-array
            Input values
        :return:
        """
        return x_values

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        """
        Return class label after unit step
        :param x_values: nd-array
            Input values
        :return:
        """
        return np.where(self.activation(x_values=self.net_input(x_values=x_values)) >= 0.0, 1, -1)

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
        self._initialize_weights(size=x_values.shape[1])
        for _ in range(self.n_iter):
            if self.shuffle:
                x_values, y_values = self._shuffle(x_values=x_values, y_values=y_values)
            cost = []
            for x_value_i, target in zip(x_values, y_values):
                cost.append(self._update_weights(x_value_i=x_value_i, target=target))
            average_cost = sum(cost) / len(y_values)
            self.cost_.append(average_cost)
        return self

    def partial_fit(self, x_values: np.ndarray, y_values: np.ndarray) -> object:
        """
        Fit training data without reinitializing the weights
        :param x_values: nd-array, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features
        :param y_values: 1d-array, shape = [n_examples]
            Target values
        :return:
        """
        if not self.weights_initialized:
            self._initialize_weights(x_values.shape[1])
        if y_values.ravel().shape[0] > 1:
            for x_value_i, target in zip(x_values, y_values):
                self._update_weights(x_value_i, target)
        else:
            self._update_weights(x_values, y_values)
        return self

    def _shuffle(self, x_values: np.ndarray, y_values: np.ndarray) -> tuple:
        """
        Shuffle training data
        :param x_values: nd-array, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features
        :param y_values: 1d-array, shape = [n_examples]
            Target values
        :return:
        """
        order_array: np.ndarray = self.np_random_state.permutation(len(x_values))
        return x_values[order_array], y_values[order_array]

    def _initialize_weights(self, size: int) -> NoReturn:
        """
        Initialize weights to small random numbers
        :param size: Weights vector size
        :return:
        """
        self.np_random_state = np.random.RandomState(self.random_state)
        self.weights_: np.ndarray = self.np_random_state.normal(
            loc=0.0, scale=0.01, size=1 + size)
        self.weights_initialized = True

    def _update_weights(self, x_value_i: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Apply Adaline learning rule to update the weights
        :param x_value_i: nd-array, shape = [1, n_features]
            Training vector
        :param target: 1d-array, shape = [1]
            Target value
        :return:
        """
        output = self.activation(x_values=self.net_input(x_values=x_value_i))
        error = (target - output)
        self.weights_[1:] += self.eta * x_value_i.dot(error)
        self.weights_[0] += self.eta * error
        cost = 0.5 * (error ** 2)
        return cost

    def net_input(self, x_values: np.ndarray) -> float:
        """
        Calculate net input
        :param x_values: nd-array, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features
        :return:
        """
        return np.dot(x_values, self.weights_[1:]) + self.weights_[0]
