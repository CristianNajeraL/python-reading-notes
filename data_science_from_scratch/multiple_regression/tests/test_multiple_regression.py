"""
Multiple Regression implementation testing module
"""

import numpy as np

from unittest import TestCase

from ..multiple_regression import MultipleRegression as mr

np.random.seed(17)


class TestMultipleRegression(TestCase):
    """
    Multiple Regression implementation testing class
    """

    values: list = [1, 2, 3]
    value_y: int = 30
    betas: list = [4, 4, 4]
    values_x: list = [[1.0, (np.random.random() < 0.5) * 1] + list(
        [float(x) for x in i]) for i in np.random.normal(size=(500, 2))]
    values_y: list = list((np.random.normal(size=500)))
    values_y = [float(x) for x in values_y]
    mr_betas: list = mr.least_squares_fit(values_x=values_x, values_y=values_y, learning_rate=1e-2,
                                          num_steps=5000, batch_size=len(values_y))

    def test_predict(self):
        """Successfully test"""
        prediction = mr.predict(values=self.values, betas=self.betas)
        self.assertEqual(prediction, 24)

    def test_error(self):
        """Successfully test"""
        error = mr.error(values=self.values, betas=self.betas, value_y=self.value_y)
        self.assertEqual(error, -6)

    def test_squared_error(self):
        """Successfully test"""
        squared_error = mr.squared_error(values=self.values, betas=self.betas, value_y=self.value_y)
        self.assertEqual(squared_error, 36)

    def test_squared_error_gradient(self):
        """Successfully test"""
        gradient = mr.squared_error_gradient(values=self.values, betas=self.betas, value_y=self.value_y)
        self.assertEqual(gradient, [-12, -24, -36])

    def test_least_squares_fit(self):
        """Successfully test"""
        self.assertAlmostEqual(self.mr_betas[0], 0.18271199141376873)
        self.assertAlmostEqual(self.mr_betas[1], -0.17492736)
        self.assertAlmostEqual(self.mr_betas[2], 0.00750139)
        self.assertAlmostEqual(self.mr_betas[3], -0.0429139)

    def test_multiple_r_squared(self):
        """Successfully test"""
        r_squared = mr.multiple_r_squared(values_x=self.values_x, values_y=self.values_y, betas=self.mr_betas)
        self.assertTrue(0.009 <= r_squared <= 0.01)
