"""
Different statistics methods implementations
"""

import math
from collections import Counter

from ..linear_algebra.vectors import Value, Vector, Vectors


class Tools:
    """
    Statistics methods implementations
    """

    @staticmethod
    def mean(vector: Vector) -> Value:
        """
        Calculate the average value of a Vector
        :param vector: Vector
        :return: The average of the Vector
        :rtype: Value
        """
        return sum(vector) / len(vector)

    @staticmethod
    def _median_odd(vector: Vector) -> Value:  # pragma: no cover
        """
        If the vector is odd, the median is the middle element
        :param vector: Vector
        :return: The median value of the Vector
        :rtype: Value
        """
        return sorted(vector)[len(vector) // 2]

    @staticmethod
    def _median_even(vector: Vector) -> Value:  # pragma: no cover
        """
        If the vector is even, the median is the average of the two middle numbers
        :param vector: Vector
        :return: The median value of the Vector
        :rtype: Value
        """
        sorted_vector = sorted(vector)
        hi_midpoint = len(vector) // 2
        return (sorted_vector[hi_midpoint - 1] + sorted_vector[hi_midpoint]) / 2

    @classmethod
    def median(cls, vector: Vector) -> Value:
        """
        Return the 'middle-most' value of Vector
        :param vector: Vector
        :return: The median value of the Vector
        :rtype: Value
        """
        size = len(vector)
        if size == 0:
            raise ValueError("Vector should have information")
        is_even = size % 2 == 0
        return cls._median_even(vector=vector) if is_even else cls._median_odd(vector=vector)

    @staticmethod
    def quantile(vector: Vector, percentile: float) -> Value:
        """
        Return the pth-percentile value of a vector
        :param vector: Vector
        :param percentile: The percentile we are looking for in the data
        :return: The value below which a given percentage of observations
            in a group of observations fall
        :rtype: Value
        """
        percentile_index = int(percentile * len(vector))
        return sorted(vector)[percentile_index]

    @staticmethod
    def mode(vector: Vector) -> Vector:
        """
        Generate a list with the mode value, because it could be more than one
        :param vector: Vector
        :return: Return a vector with the most common value(s)
        :rtype: Vector
        """
        counts = Counter(vector)
        max_count = max(counts.values())
        return [i for i, count in counts.items() if count == max_count]

    @staticmethod
    def data_range(vector: Vector) -> Value:
        """
        Difference between the largest and the smallest elements of the Vector
        :param vector: Vector
        :return: The value after the difference between the largest and smallest elements
        :rtype: Value
        """
        return max(vector) - min(vector)

    @classmethod
    def deviation_mean(cls, vector: Vector) -> Vector:
        """
        Subtracting the Vector mean from each Vector element
        :param vector: Vector
        :return: Same Vector with i-th - mean
        :rtype: Vector
        """
        vector_mean = cls.mean(vector=vector)
        return [x - vector_mean for x in vector]

    @classmethod
    def variance(cls, vector: Vector) -> Value:
        """
        The squared deviation of a random variable from its population mean or sample mean
        :param vector: Vector
        :return: The variance value
        :rtype: Value
        """
        size = len(vector)
        if size < 2:
            raise ValueError("Vector should have at least two values")
        deviations = cls.deviation_mean(vector=vector)
        return Vectors.sum_of_squares(vector=deviations) / (size - 1)

    @classmethod
    def standard_deviation(cls, vector: Vector) -> Value:
        """
        Measure of the amount of variation or dispersion of a set of values
        :param vector: Vector
        :return: The standard deviation from a random variable
        :rtype: Value
        """
        return math.sqrt(cls.variance(vector=vector))

    @classmethod
    def interquartile_range(cls, vector: Vector) -> Value:
        """
        Measure of statistical dispersion, between 75th to 25th percentile of data
        :param vector: Vector
        :return: Value that represents the interquartile range
        :rtype: Value
        """
        q25 = cls.quantile(vector=vector, percentile=0.25)
        q75 = cls.quantile(vector=vector, percentile=0.75)
        return q75 - q25

    @classmethod
    def covariance(cls, vector_x: Vector, vector_y: Vector) -> Value:
        """
        Measures the direction of the relationship between two variables
        :param vector_x: Vector
        :param vector_y: Vector
        :return: Covariance value
        :rtype: Value
        """
        de_mean_x = cls.deviation_mean(vector_x)
        de_mean_y = cls.deviation_mean(vector_y)
        dot_product_deviation_x_y = Vectors.dot(vector_x=de_mean_x, vector_y=de_mean_y)
        return dot_product_deviation_x_y / (len(vector_x) - 1)

    @classmethod
    def correlation(cls, vector_x: Vector, vector_y: Vector) -> Value:
        """
        Measures how much vector_x and vector_y vary in tandem about their means
        :param vector_x: Vector
        :param vector_y: Vector
        :return: Correlation value
        :rtype: Value
        """
        std_x = cls.standard_deviation(vector=vector_x)
        std_y = cls.standard_deviation(vector=vector_y)
        if std_x > 0 and std_y > 0:
            cov_x_y = cls.covariance(vector_x=vector_x, vector_y=vector_y)
            return cov_x_y / std_x / std_y
        return 0
