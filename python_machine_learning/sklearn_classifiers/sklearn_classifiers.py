"""
Sklearn Classifier implementation
"""

from typing import Any

import numpy as np


class SklearnClassifiers:
    """
    Class to execute multiple Sklearn classifiers
    """

    def __init__(self, estimator: Any):
        """
        Class initializer
        :param estimator: Sklearn classifier model
        """
        self.estimator = estimator

    def fit(self, x_values: np.ndarray, y_values: np.ndarray):
        """
        Fit training data
        :param x_values: nd-array, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features
        :param y_values: 1d-array, shape = [n_examples]
            Target values
        :return: Fitted sklearn classifier
        """
        self.estimator.fit(x_values, y_values)
        return self

    def score(self, x_values: np.ndarray, y_values: np.ndarray):
        """
        Return accuracy score
        :param x_values: nd-array
            Input values
        :param y_values: 1d-array, shape = [n_examples]
            Target values
        :return:
        """
        return self.estimator.score(x_values, y_values)
