"""
Testing statistics methods
"""

from unittest import TestCase

from ..tools import Tools

_VECTOR_EVEN = [1, 1, 1, 2, 3, 4, 4, 4]
_VECTOR_ODD = [1, 1, 1, 2, 3, 4, 4]
_MEAN_DEVIATION = [-1.5, -1.5, -1.5, -0.5, 0.5, 1.5, 1.5, 1.5]
_POWERED_VECTOR = [i ** 2 for i in _VECTOR_EVEN]
_ZERO_STANDARD_DEVIATION_VECTOR = [1, 1, 1, 1, 1, 1, 1, 1, 1]


class TestToolsSuccessfully(TestCase):
    """
    This class contains successfully tests for statistics methods
    """

    def test_mean(self):
        """Successfully test"""
        self.assertEqual(Tools.mean(vector=_VECTOR_EVEN), 2.5)

    def test_median_even(self):
        """Successfully test"""
        self.assertEqual(Tools.median(vector=_VECTOR_EVEN), 2.5)

    def test_median_odd(self):
        """Successfully test"""
        self.assertEqual(Tools.median(vector=_VECTOR_ODD), 2)

    def test_quantile(self):
        """Successfully test"""
        self.assertEqual(Tools.quantile(vector=_VECTOR_EVEN, percentile=0.25), 1)

    def test_mode(self):
        """Successfully test"""
        self.assertEqual(Tools.mode(vector=_VECTOR_EVEN), [1, 4])

    def test_data_range(self):
        """Successfully test"""
        self.assertEqual(Tools.data_range(vector=_VECTOR_EVEN), 3)

    def test_deviation_mean(self):
        """Successfully test"""
        self.assertEqual(Tools.deviation_mean(vector=_VECTOR_EVEN), _MEAN_DEVIATION)

    def test_variance(self):
        """Successfully test"""
        self.assertEqual(Tools.variance(vector=_VECTOR_EVEN), 2)

    def test_standard_deviation(self):
        """Successfully test"""
        self.assertEqual(Tools.standard_deviation(vector=_VECTOR_EVEN), 2**(1/2))

    def test_interquartile_range(self):
        """Successfully test"""
        self.assertEqual(Tools.interquartile_range(vector=_VECTOR_EVEN), 3)

    def test_covariance(self):
        """Successfully test"""
        self.assertEqual(Tools.covariance(vector_x=_VECTOR_EVEN, vector_y=_POWERED_VECTOR), 10)

    def test_correlation(self):
        """Successfully test"""
        corr = round(Tools.correlation(vector_x=_VECTOR_EVEN, vector_y=_POWERED_VECTOR))
        self.assertEqual(corr, 1)

    def test_correlation_zero_standard_deviation(self):
        """Successfully test"""
        vectors = [_ZERO_STANDARD_DEVIATION_VECTOR, _POWERED_VECTOR]
        corr = round(Tools.correlation(vector_x=vectors[0], vector_y=vectors[1]))
        self.assertEqual(corr, 0)


class TestToolsUnsuccessfully(TestCase):
    """
    This class contains successfully tests for statistics methods
    """

    def test_mean_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(Tools.mean(vector=_VECTOR_EVEN), 2)

    def test_median_even_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(Tools.median(vector=_VECTOR_EVEN), 2)

    def test_median_odd_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(Tools.median(vector=_VECTOR_ODD), 2.5)

    def test_median_no_data(self):
        """Unsuccessfully test"""
        with self.assertRaises(ValueError):
            Tools.median(vector=[])

    def test_quantile_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(Tools.quantile(vector=_VECTOR_EVEN, percentile=0.25), 2)

    def test_mode_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(Tools.mode(vector=_VECTOR_EVEN), [1])

    def test_data_range_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(Tools.data_range(vector=_VECTOR_EVEN), 1)

    def test_deviation_mean_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(Tools.deviation_mean(vector=_VECTOR_ODD), _MEAN_DEVIATION)

    def test_variance_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(Tools.variance(vector=_VECTOR_EVEN), 1)

    def test_variance_no_data(self):
        """Unsuccessfully test"""
        with self.assertRaises(ValueError):
            Tools.variance(vector=[])

    def test_standard_deviation_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(Tools.standard_deviation(vector=_VECTOR_EVEN), 2)

    def test_interquartile_range_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(Tools.interquartile_range(vector=_VECTOR_EVEN), 2)

    def test_covariance_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(Tools.covariance(vector_x=_VECTOR_EVEN, vector_y=_POWERED_VECTOR), 5)

    def test_correlation_wrong(self):
        """Unsuccessfully test"""
        self.assertIsNot(Tools.correlation(vector_x=_VECTOR_EVEN, vector_y=_POWERED_VECTOR), 0)
