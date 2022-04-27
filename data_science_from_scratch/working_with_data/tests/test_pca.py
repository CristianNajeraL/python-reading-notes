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
        """Successfully test"""
        self.assertEqual(PCA.de_mean(data=self.data), self.de_mean)

    def test_direction(self):
        """Successfully test"""
        direction = PCA.direction(vector_y=self.vector_y)
        self.assertTrue(0.25 <= direction[0] <= 0.27)
        self.assertTrue(0.52 <= direction[1] <= 0.54)
        self.assertTrue(0.79 <= direction[2] <= 0.81)

    def test_directional_variance(self):
        """Successfully test"""
        directional_variance = PCA.directional_variance(data=self.data, vector_y=self.vector_y)
        self.assertTrue(265.7 <= directional_variance <= 265.72)

    def test_directional_variance_gradient(self):
        """Successfully test"""
        directional_variance_gradient = PCA.directional_variance_gradient(
            data=self.data, vector_y=self.vector_y)
        self.assertTrue(262.8 <= directional_variance_gradient[0] <= 263)
        self.assertTrue(314.1 <= directional_variance_gradient[1] <= 314.3)
        self.assertTrue(365.5 <= directional_variance_gradient[2] <= 365.7)

    def test_first_principal_component(self):
        """Successfully test"""
        first_principal_component = PCA.first_principal_component(data=self.data)
        self.assertTrue(0.46 <= first_principal_component[0] <= 0.48)
        self.assertTrue(0.56 <= first_principal_component[1] <= 0.58)
        self.assertTrue(0.65 <= first_principal_component[2] <= 0.67)

    def test_project(self):
        """Successfully test"""
        project = PCA.project(vector_x=self.data[0], vector_y=self.vector_y)
        self.assertEqual(project[0], 14)
        self.assertEqual(project[1], 28)
        self.assertEqual(project[2], 42)

    def test_remove_projection_from_vector(self):
        """Successfully test"""
        remove_projection_from_vector = PCA.remove_projection_from_vector(
            vector_x=self.data[0], vector_y=self.vector_y)
        self.assertEqual(remove_projection_from_vector[0], -13)
        self.assertEqual(remove_projection_from_vector[1], -26)
        self.assertEqual(remove_projection_from_vector[2], -39)

    def test_remove_projection(self):
        """Successfully test"""
        remove_projection = PCA.remove_projection(data=self.data, vector_y=self.vector_y)
        self.assertEqual(remove_projection[0], [-13, -26, -39])
        self.assertEqual(remove_projection[1], [-28, -59, -90])
        self.assertEqual(remove_projection[2], [-43, -92, -141])

    def test_pca(self):
        """Successfully test"""
        pca = PCA.pca(data=self.data, num_components=1)
        self.assertTrue(0.46 <= pca[0][0] <= 0.48)
        self.assertTrue(0.56 <= pca[0][1] <= 0.58)
        self.assertTrue(0.65 <= pca[0][2] <= 0.67)

    def test_transform_vector(self):
        """Successfully test"""
        pca = PCA.pca(data=self.data, num_components=1)
        transform_vector = PCA.transform_vector(vector_x=self.data[0], components=pca)
        self.assertTrue(3.6 <= transform_vector[0] <= 3.62)

    def test_transform(self):
        """Successfully test"""
        pca = PCA.pca(data=self.data, num_components=1)
        transform = PCA.transform(data=self.data, components=pca)
        self.assertTrue(3.6 <= transform[0][0] <= 3.62)
        self.assertTrue(8.76 <= transform[1][0] <= 8.78)
        self.assertTrue(13.91 <= transform[2][0] <= 13.93)
