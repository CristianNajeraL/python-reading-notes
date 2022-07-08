"""
Perceptron classifier test module implementation
"""

import os
from unittest import TestCase

import numpy as np
import pandas as pd

from ..perceptron import Perceptron as Pc

path = os.path.join("https://archive.ics.uci.edu", "ml", "machine-learning-databases", "iris", "iris.data")


class TestPerceptron(TestCase):
    """
    This class contains tests for Decision classifier methods
    """

    data = pd.read_csv(path, header=None, encoding="utf-8")

    y_values = data.iloc[0:100, 4].values
    y_values = np.where(y_values == "Iris-setosa", -1, 1)

    x_values = data.iloc[0:100, [0, 2]].values

    def test_perceptron_classifier(self):
        """
        Successful perceptron classifier test
        :return:
        """
        perceptron = Pc(eta=0.1, n_iter=10)
        perceptron.fit(x_values=self.x_values, y_values=self.y_values)
        self.assertTrue(perceptron.errors_[-1] == 0)
