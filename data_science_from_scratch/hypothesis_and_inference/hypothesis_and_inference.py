"""
Hypothesis and inference methods
"""


import math
from typing import Tuple

from ..linear_algebra.vectors import Value
from ..probability.probability import Probability as P


class HypothesisAndInference:
    """
    Hypothesis and inference methods implementation
    """

    @staticmethod
    def normal_approximation_to_binomial(trials: Value, probability: Value) -> Tuple[Value, Value]:
        """
        Returns mu and sigma corresponding to a Binomial(n, p)
        :param trials: Number of attempts
        :param probability: Probability of success
        :return: Mean and variance
        :rtype: Tuple[Value, Value]
        """
        mean = probability * trials
        sigma = math.sqrt(probability * (1 - probability) * trials)
        return mean, sigma

    @staticmethod
    def normal_probability_below(value: Value, mean: Value = 0, sigma: Value = 1) -> Value:
        """
        Return the probability below the given number for normal distribution
        :param value: Random value
        :param mean: Mean
        :param sigma: Standard deviation
        :return: The probability that a normal random variable is <= x
        :rtype: Value
        """
        return P.normal_cdf(value=value, mean=mean, sigma=sigma)

    @staticmethod
    def normal_probability_above(low: Value, mean: Value = 0, sigma: Value = 1) -> Value:
        """
        Return the probability above the given number for normal distribution
        :param low: Random value
        :param mean: Mean
        :param sigma: Standard deviation
        :return: The probability that a normal random variable is <= x
        :rtype: Value
        """
        return 1 - P.normal_cdf(value=low, mean=mean, sigma=sigma)

    @staticmethod
    def normal_probability_between(low: Value, high: Value, mean: Value = 0,
                                   sigma: Value = 1) -> Value:
        """
        Return the probability within a given range for normal distribution
        :param low: Random value
        :param high: Random value
        :param mean: Mean
        :param sigma: Standard deviation
        :return: The probability that a normal random variable is <= x
        :rtype: Value
        """
        return P.normal_cdf(value=high, mean=mean, sigma=sigma) - P.normal_cdf(
            value=low, mean=mean, sigma=sigma)

    @classmethod
    def normal_probability_outside(cls, low: Value, high: Value, mean: Value = 0,
                                   sigma: Value = 1) -> Value:
        """
        Return the probability outside a given range for normal distribution
        :param low: Random value
        :param high: Random value
        :param mean: Mean
        :param sigma: Standard deviation
        :return: The probability that a normal random variable is <= x
        :rtype: Value
        """
        return 1 - cls.normal_probability_between(low=low, high=high, mean=mean, sigma=sigma)

    @staticmethod
    def normal_upper_bound(probability: Value, mean: Value = 0, sigma: Value = 1) -> Value:
        """
        Returns the z for which P(Z <= z) = probability
        :param probability: Probability
        :param mean: Mean
        :param sigma: Standard deviation
        :return: z value for given probability
        :rtype: Value
        """
        return P.inverse_normal_cdf(probability=probability, mean=mean, sigma=sigma)

    @staticmethod
    def normal_lower_bound(probability: Value, mean: Value = 0, sigma: Value = 1) -> Value:
        """
        Returns the z for which P(Z >= z) = probability
        :param probability: Probability
        :param mean: Mean
        :param sigma: Standard deviation
        :return: z value for given probability
        :rtype: Value
        """
        return P.inverse_normal_cdf(probability=1 - probability, mean=mean, sigma=sigma)

    @classmethod
    def normal_two_sided_bounds(cls, probability: Value, mean: Value = 0,
                                sigma: Value = 1) -> Tuple[Value, Value]:
        """
        Returns the symmetric (about the mean) bounds
        that contain the specified probability
        :param probability: Probability
        :param mean: Mean
        :param sigma: Standard deviation
        :return: Confidence interval
        :rtype: Tuple[Value, Value]
        """
        tail_probability = (1 - probability) / 2
        upper_bound = cls.normal_lower_bound(probability=tail_probability, mean=mean, sigma=sigma)
        lower_bound = cls.normal_upper_bound(probability=tail_probability, mean=mean, sigma=sigma)
        return lower_bound, upper_bound

    @classmethod
    def two_sided_p_value(cls, value: Value, mean: Value = 0, sigma: Value = 1) -> Value:
        """
        How likely are we to see a value at least as extreme as x (in either
        direction) if our values are from an N(mu, sigma)?
        :param value: Random value
        :param mean: Mean
        :param sigma: Standard deviation
        :return: p-value for the random value into a normal distribution
            with mean = mean and sigma = sigma
        :rtype: Value
        """
        if value >= mean:
            return 2 * cls.normal_probability_above(low=value, mean=mean, sigma=sigma)
        return 2 * cls.normal_probability_below(value=value, mean=mean, sigma=sigma)
