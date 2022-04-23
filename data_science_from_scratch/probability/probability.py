"""
Basic probability method implementations
"""

import math
import random
from collections import Counter
from typing import NoReturn

import matplotlib.pyplot as plt

from ..linear_algebra.vectors import Value


class Probability:
    """
    Probability methods implementations
    """

    _SQRT_TWO_PI = math.sqrt(2 * math.pi)

    @staticmethod
    def uniform_pdf(value: Value) -> Value:
        """
        Probability density function from an uniform distribution
        :param value: Random value
        :return: Provides a relative likelihood that the value of the
            random variable would be close to that sample
        :rtype: Value
        """
        return 1 if 0 <= value < 1 else 0

    @staticmethod
    def uniform_cdf(value: Value) -> Value:
        """
        Cumulative distribution function for uniform distribution
        :param value: Random value
        :return: The probability that a uniform random variable is <= x
        :rtype: Value
        """
        if value < 0:
            return 0
        if value < 1:
            return value
        return 1

    @classmethod
    def normal_pdf(cls, value: Value, mean: Value = 0, sigma: Value = 1) -> Value:
        """
        Probability density function from an normal distribution
        :param value: Random value
        :param mean: Mean
        :param sigma: Standard deviation
        :return: Provides a relative likelihood that the value of the
            random variable would be close to that sample
        :rtype: Value
        """
        sigma_two_pi = cls._SQRT_TWO_PI * sigma
        sigma_pow = sigma ** 2
        mean_de = value - mean
        return math.exp(-mean_de ** 2 / 2 / sigma_pow) / sigma_two_pi

    @staticmethod
    def normal_cdf(value: Value, mean: Value = 0, sigma: Value = 1) -> Value:
        """
        Cumulative distribution function for normal distribution
        :param value: Random value
        :param mean: Mean
        :param sigma: Standard deviation
        :return: The probability that a normal random variable is <= x
        :rtype: Value
        """
        return (1 + math.erf((value - mean) / math.sqrt(2) / sigma)) / 2

    @classmethod
    def inverse_normal_cdf(cls, probability: Value, mean: Value = 0,
                           sigma: Value = 1, tolerance: Value = 0.00001) -> Value:
        """
        Finds the Z that's close enough to the desired probability
        :param probability: Probability
        :param mean: Mean
        :param sigma: Standard deviation
        :param tolerance: Tolerance
        :return:
        """
        if mean != 0 or sigma != 1:
            return mean + sigma * cls.inverse_normal_cdf(probability=probability,
                                                         tolerance=tolerance)
        low_z: Value = -10.0
        hi_z: Value = 10.0
        while hi_z - low_z > tolerance:
            mid_z: Value = (low_z + hi_z) / 2
            mid_p = cls.normal_cdf(value=mid_z)
            if mid_p < probability:
                low_z = mid_z
            else:
                hi_z = mid_z
        return mid_z

    @staticmethod
    def bernoulli_trial(probability: Value) -> int:
        """
        Returns 1 with probability p and 0 with probability 1-p
        :param probability: Probability
        :return: 1 or 0 depending on the probability threshold
        :rtype: int
        """
        return 1 if random.random() < probability else 0

    @classmethod
    def binomial(cls, trials: int, probability: Value) -> int:
        """
        Returns the sum of n bernoulli(p) trials
        :param trials: Number of attempts that the experiment is going to run
        :param probability: Probability
        :return: Sum of the successfully events
        :rtype: int
        """
        return sum(cls.bernoulli_trial(probability=probability) for _ in range(trials))

    @classmethod
    def binomial_histogram(cls, probability: Value, trials: int,
                           num_of_experiments: int) -> NoReturn:   # pragma: no cover
        """

        :param probability:
        :param trials:
        :param num_of_experiments:
        :return:
        """
        data = [cls.binomial(trials=trials, probability=probability)
                for _ in range(num_of_experiments)]
        histogram = Counter(data)
        plt.bar([x - 0.4 for x in histogram.keys()],
                [v / num_of_experiments for v in histogram.values()],
                0.8, color=0.75)
        mean = probability * trials
        sigma = math.sqrt(trials * probability * (1 - probability))
        x_values = range(min(data), max(data) + 1)
        y_values = [cls.normal_cdf(value=i + 0.5, mean=mean, sigma=sigma) -
                    cls.normal_cdf(value=i - 0.5, mean=mean, sigma=sigma) for i in x_values]
        plt.plot(x_values, y_values)
        plt.title("Binomial Distribution vs. Normal Approximation")
        plt.show()
