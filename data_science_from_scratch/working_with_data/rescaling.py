"""
How to create rescale data
"""

from typing import List, Tuple

from ..linear_algebra.vectors import Vector, Vectors
from ..statistics.tools import Tools


class Rescaling:
    """
    Rescale data method implementation
    """

    @staticmethod
    def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
        """
        Returns the mean and standard deviation for each column
        :param data: Input data
        :return: Mean and standard deviation for each column
        :rtype: Tuple[Vector, Vector]
        """
        size = len(data[0])
        means = Vectors.vector_mean(vectors=data)
        standard_deviations = [
            Tools.standard_deviation(vector=[vector[i] for vector in data]) for i in range(size)
        ]
        return means, standard_deviations

    @classmethod
    def rescale(cls, data: List[Vector]) -> List[Vector]:
        """
        Rescales the input data so that each position has
            mean 0 and standard deviation 1. (Leaves a position
            as is if its standard deviation is 0.)
        :param data: Input data
        :return: Rescaled data
        :rtype: List[Vector]
        """
        size = len(data[0])
        means, standard_deviations = cls.scale(data=data)
        rescaled = [vector[:] for vector in data]
        for vector in rescaled:
            for i in range(size):
                if standard_deviations[i] > 0:
                    vector[i] = (vector[i] - means[i]) / standard_deviations[i]
        return rescaled
