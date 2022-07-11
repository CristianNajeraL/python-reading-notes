"""
Logistic Regression implementation
"""

import numpy as np


class LogisticRegressionGD:  # pylint: disable=E1101 disable=R0902 disable=R0801
    """
    Logistic Regression using gradient descent

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
        Compute logistic sigmoid activation
        :param x_values: nd-array
            Input values
        :return:
        """
        return 1.0 / (1.0 + np.exp(-np.clip(x_values, -250, 250)))

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        """
        Return class label after unit step
        :param x_values: nd-array
            Input values
        :return:
        """
        return np.where(self.net_input(x_values=x_values) >= 0.0, 1, 0)

    def net_input(self, x_values: np.ndarray) -> float:
        """
        Calculate net input
        :param x_values: nd-array, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features
        :return:
        """
        return np.dot(x_values, self.weights_[1:]) + self.weights_[0]

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
        self.np_random_state = np.random.RandomState(self.random_state)
        self.weights_: np.ndarray = self.np_random_state.normal(
            loc=0.0, scale=0.01, size=1 + x_values.shape[1])
        for _ in range(self.n_iter):
            net_input = self.net_input(x_values=x_values)
            output = self.activation(x_values=net_input)
            errors = (y_values - output)
            self.weights_[1:] += self.eta * x_values.T.dot(errors)
            self.weights_[0] += self.eta * errors.sum()
            cost = (
                -y_values.dot(np.log(output)) - (
                    (1 - y_values).dot(np.log(1 - output))
                )
            )
            self.cost_.append(cost)
        return self
