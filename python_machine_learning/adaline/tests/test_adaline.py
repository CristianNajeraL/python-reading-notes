"""
Adaline classifier test module implementation
"""

import os
from unittest import TestCase

import numpy as np
import pandas as pd

from ..adaline import Adaline as Ad

path = os.path.join("https://archive.ics.uci.edu", "ml", "machine-learning-databases", "iris", "iris.data")


class TestAdaline(TestCase):
    """
    This class contains tests for Adaline classifier methods
    """
    data = pd.read_csv(path, header=None, encoding="utf-8")

    y_values = data.iloc[0:100, 4].values
    y_values = np.where(y_values == "Iris-setosa", -1, 1)

    x_values = data.iloc[0:100, [0, 2]].values
    x_std = np.copy(x_values)
    x_std[:, 0] = (x_values[:, 0] - x_values[:, 0].mean()) / x_values[:, 0].std()
    x_std[:, 1] = (x_values[:, 1] - x_values[:, 1].mean()) / x_values[:, 1].std()

    def test_adaline_classifier(self):
        """
        Successful perceptron classifier test
        :return:
        """
        ada = Ad(eta=0.01, n_iter=20).fit(self.x_std, self.y_values)
        self.assertTrue(all(ada.predict(x_values=self.x_std) == self.y_values))

    def test_partial_fit(self):
        """
        Successful perceptron classifier test
        :return:
        """
        ada = Ad(eta=0.01, n_iter=20).fit(self.x_std, self.y_values)
        ada.partial_fit(self.x_std[0, :], self.y_values[0])
        self.assertAlmostEqual(ada.weights_[0], -0.00369169)
        self.assertAlmostEqual(ada.weights_[1], -0.16544708)
        self.assertAlmostEqual(ada.weights_[2], 1.09308945)

    def test_partial_fit_line_95(self):
        """
        Successful perceptron classifier test
        :return:
        """
        ada = Ad(eta=0.01, n_iter=20).fit(self.x_std, self.y_values)
        ada.weights_initialized = False
        ada.partial_fit(self.x_std[0:10, :], self.y_values[0:10])
        self.assertAlmostEqual(ada.weights_[0], -0.08660881)
        self.assertAlmostEqual(ada.weights_[1], 0.06576494)
        self.assertAlmostEqual(ada.weights_[2], 0.09389597)
