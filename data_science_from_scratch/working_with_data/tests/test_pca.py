"""
Testing principal component analysis implementation
"""

from unittest import TestCase

from ..pca import PCA


class TestPCA(TestCase):
    """
    This class contains tests for principal component analysis implementation
    """

    data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    de_mean = [
        [-3, -3, -3],
        [0, 0, 0],
        [3, 3, 3]
    ]

    vector_y = [1, 2, 3]

    direction = [0.26, 0.53, 0.80]

    directional_variance = 265.71

    def test_de_mean(self):
        self.assertEqual(PCA.de_mean(data=self.data), self.de_mean)

    def test_direction(self):
        direction = PCA.direction(vector_y=self.vector_y)
        self.assertTrue(0.25 <= direction[0] <= 0.27)
        self.assertTrue(0.52 <= direction[1] <= 0.54)
        self.assertTrue(0.79 <= direction[2] <= 0.81)

    def test_directional_variance(self):
        directional_variance = PCA.directional_variance(data=self.data, vector_y=self.vector_y)
        self.assertTrue(265.7 <= directional_variance <= 265.72)

    # def test_directional_variance_gradient(self):
    #     self.fail()
    #
    # def test_first_principal_component(self):
    #     self.fail()
    #
    # def test_project(self):
    #     self.fail()
    #
    # def test_remove_projection_from_vector(self):
    #     self.fail()
    #
    # def test_remove_projection(self):
    #     self.fail()
    #
    # def test_pca(self):
    #     self.fail()
    #
    # def test_transform_vector(self):
    #     self.fail()
    #
    # def test_transform(self):
    #     self.fail()
