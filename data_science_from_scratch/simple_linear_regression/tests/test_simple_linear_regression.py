"""
Test module for Simple Linear Regression Implementation
"""

from unittest import TestCase

from ..simple_linear_regression import SimpleLinearRegression as slr


class TestSimpleLinearRegression(TestCase):
    """
    This class contains tests for probability methods
    """

    inputs: list = list(range(-100, 100, 10))
    outputs: list = list(3 * i - 5 for i in inputs)

    def test_predict(self):
        """Successfully test"""
        self.assertEqual(slr.predict(alpha=1, beta=0, value=2), 1)

    def test_error(self):
        """Successfully test"""
        self.assertEqual(slr.error(alpha=1, beta=0, value=2, actual=0), 1)

    def test_sum_of_squared_errors(self):
        """Successfully test"""
        squared_error = slr.sum_of_squared_errors(
            alpha=1, beta=0, inputs=self.inputs, outputs=self.outputs)
        self.assertEqual(squared_error, 607320)

    def test_least_squares_fit(self):
        """Successfully test"""
        alpha, beta = slr.least_squares_fit(inputs=self.inputs, outputs=self.outputs)
        self.assertTrue(-5.1 <= alpha <= -4.9)
        self.assertTrue(2.9 <= beta <= 3.1)

    def test_total_sum_of_squares(self):
        """Successfully test"""
        self.assertEqual(slr.total_sum_of_squares(vector=self.outputs), 598500.0)

    def test_r_squared(self):
        """Successfully test"""
        alpha, beta = slr.least_squares_fit(inputs=self.inputs, outputs=self.outputs)
        r_squared = slr.r_squared(alpha=alpha, beta=beta, inputs=self.inputs, outputs=self.outputs)
        self.assertEqual(r_squared, 1.0)

    def test_gradient_descent_fit(self):
        """Successfully test"""
        alpha, beta = slr.gradient_descent_fit(inputs=self.inputs, outputs=self.outputs)
        self.assertTrue(-5.1 <= alpha <= -4.9)
        self.assertTrue(2.9 <= beta <= 3.1)
