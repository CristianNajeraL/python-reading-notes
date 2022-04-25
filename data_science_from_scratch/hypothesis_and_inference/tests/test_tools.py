"""
Testing hypothesis and inference methods
"""

from unittest import TestCase

from ..hypothesis_and_inference import HypothesisAndInference as hyp_inf


class TestHypothesisAndInference(TestCase):  # pragma: no cover
    """
    This class contains tests for hypothesis and inference methods
    """

    def test_normal_approximation_to_binomial(self):
        """Successfully test"""
        mean, sigma = hyp_inf.normal_approximation_to_binomial(trials=1000, probability=0.5)
        self.assertEqual(mean, 500)
        self.assertEqual(round(sigma, 2), 15.81)

    def test_normal_approximation_to_binomial_wrong(self):
        """Unsuccessfully test"""
        mean, sigma = hyp_inf.normal_approximation_to_binomial(trials=1000, probability=0.5)
        self.assertIsNot(mean, 450)
        self.assertIsNot(round(sigma, 2), 10)

    def test_normal_probability_below(self):
        """Successfully test"""
        self.assertEqual(hyp_inf.normal_probability_below(value=0), 0.5)

    def test_normal_probability_below_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(hyp_inf.normal_probability_below(value=0), 1)

    def test_normal_probability_above(self):
        """Successfully test"""
        self.assertEqual(hyp_inf.normal_probability_above(low=0), 0.5)

    def test_normal_probability_above_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(hyp_inf.normal_probability_above(low=0), 1)

    def test_normal_probability_between(self):
        """Successfully test"""
        probability = hyp_inf.normal_probability_between(low=0, high=1.64)
        self.assertEqual(round(probability, 2), 0.45)

    def test_normal_probability_between_wrong(self):
        """Unsuccessfully test"""
        probability = hyp_inf.normal_probability_between(low=0, high=1.64)
        self.assertIsNot(round(probability, 2), 1)

    def test_normal_probability_outside(self):
        """Successfully test"""
        probability = hyp_inf.normal_probability_outside(low=0, high=1.64)
        self.assertEqual(round(probability, 2), 0.55)

    def test_normal_probability_outside_wrong(self):
        """Unsuccessfully test"""
        probability = hyp_inf.normal_probability_outside(low=0, high=1.64)
        self.assertIsNot(round(probability, 2), 1)

    def test_normal_upper_bound(self):
        """Successfully test"""
        z_value = hyp_inf.normal_upper_bound(probability=0.95)
        self.assertEqual(round(z_value, 2), 1.64)

    def test_normal_upper_bound_wrong(self):
        """Unsuccessfully test"""
        z_value = hyp_inf.normal_upper_bound(probability=0.95)
        self.assertIsNot(round(z_value, 2), 1.5)

    def test_normal_lower_bound(self):
        """Successfully test"""
        z_value = hyp_inf.normal_upper_bound(probability=0.05)
        self.assertEqual(round(z_value, 2), -1.64)

    def test_normal_lower_bound_wrong(self):
        """Unsuccessfully test"""
        z_value = hyp_inf.normal_upper_bound(probability=0.05)
        self.assertIsNot(round(z_value, 2), 1.5)

    def test_normal_two_sided_bounds(self):
        """Successfully test"""
        low, high = hyp_inf.normal_two_sided_bounds(0.9)
        self.assertEqual(round(low, 2), -1.64)
        self.assertEqual(round(high, 2), 1.64)

    def test_normal_two_sided_bounds_wrong(self):
        """Unsuccessfully test"""
        low, high = hyp_inf.normal_two_sided_bounds(0.9)
        self.assertIsNot(round(low, 2), -1.5)
        self.assertIsNot(round(high, 2), 1.5)

    def test_two_sided_p_value(self):
        """Successfully test"""
        self.assertEqual(hyp_inf.two_sided_p_value(0), 1)

    def test_two_sided_p_value_lower(self):
        """Successfully test"""
        self.assertEqual(round(hyp_inf.two_sided_p_value(-1), 2), 0.32)

    def test_two_sided_p_value_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(hyp_inf.two_sided_p_value(0), 0)
