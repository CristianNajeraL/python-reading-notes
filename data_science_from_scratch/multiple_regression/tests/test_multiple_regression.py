"""
Multiple Regression implementation testing module
"""

from unittest import TestCase

from ..multiple_regression import MultipleRegression as mr


class TestMultipleRegression(TestCase):
    """
    Multiple Regression implementation testing class
    """

    betas: list = list(range(5))
    values: list = list(i ** 2 for i in betas)

    def test_predict(self):
        """Successfully test"""
        prediction = mr.predict(values=self.values, betas=self.betas)
        self.assertEqual(prediction, 100)
