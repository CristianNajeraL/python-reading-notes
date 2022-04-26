"""
Testing Gradient Descent methods
"""

import random
from unittest import TestCase

from ..gradient_descent import GradientDescent as gd


class TestGradientDescent(TestCase):
    """
    This class contains tests for Gradient Descent methods
    """

    vector = [random.uniform(-10, 10) for i in range(3)]
    inputs = [(x, 20 * x + 5) for x in range(-50, 50)]
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

    def test_reduce_distance(self):
        """Successfully test"""
        distance = gd.reduce_distance(steps=500, vector=self.vector, step_size=-0.01)
        self.assertTrue(-0.01 <= distance <= 0.01)

    def test_reduce_linear_gradient(self):
        """Successfully test"""
        slope, intercept = gd.reduce_linear_gradient(steps=5000, learning_rate=0.001,
                                                     inputs=self.inputs, theta=self.theta,
                                                     batches=False)
        self.assertTrue(19.9 <= slope <= 20.1)
        self.assertTrue(4.9 <= intercept <= 5.1)

    def test_reduce_linear_gradient_batches(self):
        """Successfully test"""
        slope, intercept = gd.reduce_linear_gradient(steps=5000, learning_rate=0.001,
                                                     inputs=self.inputs, theta=self.theta)
        self.assertTrue(19.9 <= slope <= 20.1)
        self.assertTrue(4.9 <= intercept <= 5.1)
