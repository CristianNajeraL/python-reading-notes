"""
How to implement principal component analysis
"""

from typing import List

import tqdm

from ..gradient_descent import GradientDescent as gd
from ..linear_algebra.vectors import Vector
from ..linear_algebra.vectors import Vectors as v


class PCA:
    """
    Principal component analysis implementation
    """

    @staticmethod
    def de_mean(data: List[Vector]) -> List[Vector]:
        """
        Recenter data to have mean 0 in every column (dimension)
        :param data: Input data
        :return: Input data centered to zero
        :rtype: List[Vector]
        """
        mean = v.vector_mean(vectors=data)
        return [v.subtract(vector_x=vector, vector_y=mean) for vector in data]

    @staticmethod
    def direction(vector_y: Vector) -> Vector:
        """
        Return the direction of the vector
        :param vector_y: Input vector
        :return: Vector
        :rtype: Vector
        """
        magnitude = v.magnitude(vector=vector_y)
        return [i / magnitude for i in vector_y]

    @classmethod
    def directional_variance(cls, data: List[Vector], vector_y: Vector) -> float:
        """
        Return the variance of x in the direction of w
        :param data: Input data
        :param vector_y: Input weights
        :return: float
        :rtype: float
        """
        direction = cls.direction(vector_y=vector_y)
        return sum(v.dot(vector_x=vector, vector_y=direction) ** 2 for vector in data)

    @classmethod
    def directional_variance_gradient(cls, data: List[Vector], vector_y: Vector) -> Vector:
        """
        The gradient of directional variance with respect to w
        :param data: Input data
        :param vector_y: Input vector
        :return: Vector
        :rtype: Vector
        """
        direction = cls.direction(vector_y=vector_y)
        return [sum(2 * v.dot(vector_x=vector, vector_y=direction) * vector[i] for vector in data)
                for i in range(len(vector_y))]

    @classmethod
    def first_principal_component(cls, data: List[Vector], steps: int = 100,
                                  step_size: float = 0.1) -> Vector:
        """
        First principal component
        :param data: Input data
        :param steps: Number of attempts
        :param step_size: Size of the 'correction' step
        :return: Vector
        :rtype: Vector
        """
        guess = [1.0 for _ in data[0]]
        with tqdm.trange(steps) as loop:
            for _ in loop:
                directional_variance = cls.directional_variance(data=data, vector_y=guess)
                gradient = cls.directional_variance_gradient(data=data, vector_y=guess)
                guess = gd.gradient_step(vector=guess, gradient=gradient, step_size=step_size)
                loop.set_description(f"Directional Variance: {directional_variance:.3f}")
        return cls.direction(vector_y=guess)

    @staticmethod
    def project(vector_x: Vector, vector_y: Vector) -> Vector:
        """
        Return the projection of vector_x onto the direction vector_y
        :param vector_x: Input vector
        :param vector_y: Input vector
        :return: The projection of vector_x onto the direction vector_y
        :rtype: Vector
        """
        projection_length = v.dot(vector_x=vector_x, vector_y=vector_y)
        return v.scalar_multiply(scalar=projection_length, vector=vector_y)

    @classmethod
    def remove_projection_from_vector(cls, vector_x: Vector, vector_y: Vector) -> Vector:
        """
        Projects vector_x onto vector_y and subtracts the result from vector_x
        :param vector_x: Input vector
        :param vector_y: Input vector
        :return: Projects vector_x onto vector_y and subtracts the result from vector_x
        :rtype: Vector
        """
        return v.subtract(vector_x=vector_x, vector_y=cls.project(
            vector_x=vector_x, vector_y=vector_y))

    @classmethod
    def remove_projection(cls, data: List[Vector], vector_y: Vector) -> List[Vector]:
        """
        Remove the projection for every single vector into the input data
        :param data: Input data
        :param vector_y: Input vector
        :return: Remove the projection for every single vector into the input data
        :rtype: List[Vector]
        """
        return [cls.remove_projection_from_vector(vector_x=vector_x, vector_y=vector_y)
                for vector_x in data]

    @classmethod
    def pca(cls, data: List[Vector], num_components: int) -> List[Vector]:
        """
        Execute principal component analysis
        :param data: Input data
        :param num_components: Number of components
        :return: Input data with a number of components dimensionality
        :rtype: List[Vector]
        """
        components: List[Vector] = []
        for _ in range(num_components):
            components.append(cls.first_principal_component(data=data))
            data = cls.remove_projection(data=data, vector_y=components[-1])
        return components

    @staticmethod
    def transform_vector(vector_x: Vector, components: List[Vector]) -> Vector:
        """
        Reduce a vector
        :param vector_x: Input vector
        :param components: Principal components
        :return: Vector
        :rtype: Vector
        """
        return [v.dot(vector_x=vector_x, vector_y=vector_y) for vector_y in components]

    @classmethod
    def transform(cls, data: List[Vector], components: List[Vector]) -> List[Vector]:
        """
        Reduce a complete matrix
        :param data: Input data
        :param components: Principal components
        :return: Matrix reduced
        :rtype: List[Vector]
        """
        return [cls.transform_vector(vector_x=vector_x, components=components) for vector_x in data]
