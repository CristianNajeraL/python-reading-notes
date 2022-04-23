"""
Testing probability methods
"""

from unittest import TestCase, mock

from ..probability import Probability as p


class TestProbability(TestCase):
    """
    This class contains tests for probability methods
    """

    def test_uniform_pdf(self):
        """Successfully test"""
        self.assertEqual(p.uniform_pdf(value=0.5), 1)

    def test_uniform_pdf_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(p.uniform_pdf(value=0.5), 0)

    def test_uniform_cdf(self):
        """Successfully test"""
        self.assertEqual(p.uniform_cdf(value=0.5), 0.5)

    def test_uniform_cdf_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(p.uniform_cdf(value=0.5), 0)

    def test_uniform_cdf_zero(self):
        """Successfully test"""
        self.assertEqual(p.uniform_cdf(value=-1), 0)

    def test_uniform_cdf_zero_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(p.uniform_cdf(value=-1), 1)

    def test_uniform_cdf_one(self):
        """Successfully test"""
        self.assertEqual(p.uniform_cdf(value=1.2), 1)

    def test_uniform_cdf_one_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(p.uniform_cdf(value=1.2), 0)

    def test_normal_pdf(self):
        """Successfully test"""
        value: float = round(p.normal_pdf(0), 2)
        self.assertEqual(value, 0.4)

    def test_normal_pdf_wrong(self):
        """Unsuccessfully test"""
        value: float = round(p.normal_pdf(0), 2)
        self.assertIsNot(value, 0.5)

    def test_normal_cdf(self):
        """Successfully test"""
        value: float = round(p.normal_cdf(0), 2)
        self.assertEqual(value, 0.5)

    def test_normal_cdf_wrong(self):
        """Unsuccessfully test"""
        value: float = round(p.normal_cdf(0), 2)
        self.assertIsNot(value, 0.4)

    def test_inverse_normal_cdf(self):
        """Successfully test"""
        value: float = round(p.inverse_normal_cdf(0.95), 2)
        self.assertEqual(value, 1.64)

    def test_inverse_normal_cdf_no_mean_zero(self):
        """Successfully test"""
        value: float = round(p.inverse_normal_cdf(0.95, 1, 2), 2)
        self.assertEqual(value, 4.29)

    def test_inverse_normal_cdf_wrong(self):
        """Unsuccessfully test"""
        value: float = round(p.inverse_normal_cdf(0.95), 2)
        self.assertIsNot(value, 2)

    @mock.patch('random.random')
    def test_bernoulli_trial(self, mock_choice):
        """Successfully test"""
        mock_choice.side_effect = [0.5]
        self.assertEqual(p.bernoulli_trial(0.9), 1)

    @mock.patch('random.random')
    def test_bernoulli_trial_wrong(self, mock_choice):
        """Unsuccessfully test"""
        mock_choice.side_effect = [0.5]
        self.assertIsNot(p.bernoulli_trial(0.4), 1)

    @mock.patch('random.random')
    def test_binomial(self, mock_choice):
        """Successfully test"""
        trials: int = 10
        mock_choice.side_effect = [i / trials for i in range(trials)]
        self.assertEqual(p.binomial(trials=trials, probability=0.7), 7)

    @mock.patch('random.random')
    def test_binomial_wrong(self, mock_choice):
        """Unsuccessfully test"""
        trials: int = 10
        mock_choice.side_effect = [i / trials for i in range(trials)]
        self.assertIsNot(p.binomial(trials=trials, probability=0.7), 6)
