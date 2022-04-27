"""
Testing rescaling method
"""

from unittest import TestCase

from ..rescaling import Rescaling as r


class TestRescaling(TestCase):
    """
    This class contains tests for rescaling method
    """

    vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]

    def test_scale(self):
        """Successfully test"""
        means, standard_deviations = r.scale(data=self.vectors)
        self.assertEqual(means, [-1, 0, 1])
        self.assertEqual(standard_deviations, [2, 1, 0])

    def test_rescale(self):
        """Successfully test"""
        means, standard_deviations = r.scale(data=r.rescale(data=self.vectors))
        self.assertEqual(means, [0, 0, 1])
        self.assertEqual(standard_deviations, [1, 1, 0])
