"""
Multiple Regression implementation module
"""

from ..linear_algebra.vectors import Value, Vector
from ..linear_algebra.vectors import Vectors as v


class MultipleRegression:
    """
    Multiple regression implementation class
    """

    @staticmethod
    def predict(values: Vector, betas: Vector) -> Value:
        """
        Takes an input and generates a prediction based on the calculated coefficients
        :param betas: Coefficients to make the prediction
        :param values: Input values
        :return: Model prediction
        """
        return v.dot(vector_x=values, vector_y=betas)
