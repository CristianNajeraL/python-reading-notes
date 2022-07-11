"""
Logistic Regression test module implementation
"""

from unittest import TestCase

from sklearn import datasets
from sklearn.model_selection import train_test_split

from ..logistic_regression_gd import LogisticRegressionGD as Lr


class TestLogisticRegressionGD(TestCase):
    """
    This class contains tests for Logistic Regression methods
    """

    iris = datasets.load_iris()
    X_data = iris.data[:, [2, 3]]
    y_data = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=17, stratify=y_data
    )
    X_train_01 = X_train[(y_train == 0) | (y_train == 1)]
    y_train_01 = y_train[(y_train == 0) | (y_train == 1)]

    def test_logistic_regression(self):
        """
        Successful perceptron classifier test
        :return:
        """
        logit = Lr(eta=0.05, n_iter=1000, random_state=17)
        logit.fit(x_values=self.X_train_01, y_values=self.y_train_01)
        self.assertEqual((logit.predict(x_values=self.X_train_01) != self.y_train_01).sum(), 0)
