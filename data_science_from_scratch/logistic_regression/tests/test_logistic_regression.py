"""
Logistic Regression implementation testing module
"""

import numpy as np

from ..logistic_regression import LogisticRegression as Lr

from ...linear_algebra.vectors import Value, Vector
from ...machine_learning.machine_learning import MachineLearning as Ml
from ...working_with_data.rescaling import Rescaling as Re
from typing import List, Tuple
from unittest import TestCase


class TestLogisticRegression(TestCase):
    """
    Logistic Regression implementation testing class
    """

    np.random.seed(17)

    zero_experience: np.ndarray = np.random.uniform(low=1, high=3, size=(100,))
    zero_salary: np.ndarray = np.random.uniform(low=40000, high=60000, size=(100,))
    zeros: np.ndarray = np.zeros(100)

    one_experience: np.ndarray = np.random.uniform(low=2.5, high=7, size=(100,))
    one_salary: np.ndarray = np.random.uniform(low=50000, high=120000, size=(100,))
    ones: np.ndarray = np.ones(100)

    experience: np.ndarray = np.concatenate((zero_experience, one_experience))
    salary: np.ndarray = np.concatenate((zero_salary, one_salary))
    paid_account: np.ndarray = np.concatenate((zeros, ones))

    data = [list(i) for i in np.array([np.ones(200), experience, salary, paid_account]).T]
    np.random.shuffle(data)
    values_x: list = [[float(j) for j in i[:3]] for i in data]
    values_y: list = [float(i[3]) for i in data]

    rescaled_x: List[Vector] = Re.rescale(data=values_x)
    x_train, x_test, y_train, y_test = Ml.train_test_split(data_x=rescaled_x, data_y=values_y, test_share=0.33)

    learning_rate: Value = 0.01
    beta: List = [float(i) for i in np.random.normal(size=3)]

    result: Tuple = Lr.fit_model(beta=beta, x_train=x_train, y_train=y_train, steps=1000)

    def test_fit_model(self):
        """Successfully test"""
        self.assertTrue(self.result[1] < 10)
        self.assertTrue(self.result[0][0] < 10)
        self.assertTrue(self.result[0][1] < 10)
        self.assertTrue(self.result[0][2] < 10)

    def test_precision_recall(self):
        """Successfully test"""
        # Lr.precision_recall()
        print(self.result)
