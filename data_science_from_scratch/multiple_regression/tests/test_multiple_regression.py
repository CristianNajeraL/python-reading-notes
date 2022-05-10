"""
Multiple Regression implementation testing module
"""
import random
from unittest import TestCase

import numpy as np

from ...statistics import Tools as t
from ..multiple_regression import MultipleRegression as mr


class TestMultipleRegression(TestCase):
    """
    Multiple Regression implementation testing class
    """
    np.random.seed(17)

    size: int = 50
    values: list = [1, 2, 3]
    value_y: int = 30
    betas: list = [4, 4, 4]
    values_x: list = [[1.0, (np.random.random() < 0.5) * 1] + list(
        [float(x) for x in i]) for i in np.random.random(size=(size, 2))]
    values_y: list = list((np.random.random(size=size)))
    values_y = [float(x) for x in values_y]
    mr_betas: list = mr.least_squares_fit(values_x=values_x, values_y=values_y, learning_rate=1e-4,
                                          num_steps=500, batch_size=len(values_y))
    close_to_100: list = [99.5 + random.random() for _ in range(101)]
    far_from_100: list = (
        [99.5 + random.random()] +
        [random.random() for _ in range(50)] +
        [200 + random.random() for _ in range(50)]
    )
    bootstrap_betas: list = mr.bootstrap_statistic(data=list(zip(values_x, values_y)),
                                                   stats_fn=mr.estimate_sample_beta, num_samples=100)

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
        self.assertEqual(len(self.mr_betas), 4)
        self.assertTrue(all(i < 1 for i in self.mr_betas))

    def test_multiple_r_squared(self):
        """Successfully test"""
        r_squared = mr.multiple_r_squared(values_x=self.values_x, values_y=self.values_y, betas=self.mr_betas)
        self.assertTrue(r_squared < 0)

    def test_bootstrap_statistic(self):
        medians_close = mr.bootstrap_statistic(self.close_to_100, t.median, 100)
        medians_far = mr.bootstrap_statistic(self.far_from_100, t.median, 100)
        self.assertTrue(t.standard_deviation(medians_close) < 1)
        self.assertTrue(t.standard_deviation(medians_far) > 90)

    def test_estimate_sample_beta(self):
        bootstrap_standard_errors: list = [t.standard_deviation([beta[i] for beta in self.bootstrap_betas]) for i in
                                           range(4)]
        self.assertEqual(len(bootstrap_standard_errors), 4)
        self.assertTrue(all(i < 1 for i in bootstrap_standard_errors))

    def test_p_value(self):
        bootstrap_standard_errors: list = [t.standard_deviation([beta[i] for beta in self.bootstrap_betas]) for i in
                                           range(4)]
        p_value_0 = mr.p_value(beta_hat_j=self.mr_betas[0], sigma_hat_j=bootstrap_standard_errors[0])
        p_value_1 = mr.p_value(beta_hat_j=self.mr_betas[1], sigma_hat_j=bootstrap_standard_errors[1])
        p_value_2 = mr.p_value(beta_hat_j=self.mr_betas[2] - 1, sigma_hat_j=bootstrap_standard_errors[2])
        p_value_3 = mr.p_value(beta_hat_j=self.mr_betas[3] - 1, sigma_hat_j=bootstrap_standard_errors[3])
        self.assertTrue(p_value_0 < 1)
        self.assertTrue(p_value_1 < 1)
        self.assertTrue(p_value_2 < 1)
        self.assertTrue(p_value_3 < 1)

    def test__ridge_penalty(self):
        self.assertTrue(mr._ridge_penalty(betas=self.mr_betas, alpha=0.5) < 1)

    def test_squared_error_ridge(self):
        self.assertTrue(mr.squared_error_ridge(values_x=self.values_x[0], value_y=self.values_y[0], betas=self.mr_betas,
                                               alpha=0.5) < 2)

    def test__ridge_penalty_gradient(self):
        self.assertTrue(len(mr._ridge_penalty_gradient(betas=self.mr_betas, alpha=0.5)) == 4)

    def test_squared_error_ridge_penalty(self):
        self.assertTrue(len(mr.squared_error_ridge_penalty(values_x=self.values_x[0], value_y=self.values_y[0],
                                                           betas=self.mr_betas, alpha=0.5)) == 4)

    def test_least_squares_fit_ridge(self):
        self.assertTrue(len(mr.least_squares_fit_ridge(values_x=self.values_x, values_y=self.values_y, alpha=0.5,
                                                       num_steps=500, batch_size=100)) == 4)
