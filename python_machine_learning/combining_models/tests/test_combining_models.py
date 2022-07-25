"""
Combining models test module implementation
"""

from unittest import TestCase

from ..combining_mdoels import CombiningModels as Cm


class TestDimensionalityReduction(TestCase):
    """
    This class contains tests for Dimensionality reduction methods
    """

    def test_ensemble_pipeline(self):
        """
        Successfully test
        :return:
        """
        self.assertTrue(Cm.ensemble_pipeline() >= 0.9)

    def test_bagging(self):
        """
        Successfully test
        :return:
        """
        self.assertTrue(Cm().bagging()[1] >= 0.8)

    def test_boosting(self):
        """
        Successfully test
        :return:
        """
        self.assertTrue(Cm().boosting()[1] >= 0.8)
