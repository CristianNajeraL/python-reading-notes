"""
Basic machine learning implementations
"""

import random
from typing import List, Tuple, TypeVar

X = TypeVar('X')
Y = TypeVar('Y')


class MachineLearning:
    """
    Machine learning concepts
    """

    @staticmethod
    def split_data(data: List[X], probability: float) -> Tuple[List[X], List[X]]:
        """
        Split data into fractions [probability, 1 - probability]
        :param data: Input data
        :param probability: probability
        :return: Two sets of data
        """
        data = data[:]
        random.shuffle(x=data)
        cut = int(len(data) * probability)
        return data[:cut], data[cut:]

    @classmethod
    def train_test_split(cls, data_x: List[X], data_y: List[Y],
                         test_share: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
        """
        Split X and Y data into fractions
        :param data_x: Input data
        :param data_y: Input data
        :param test_share: Probability
        :return: Four sets of data
        """
        indexes = list(i for i in range(len(data_x)))
        train_indexes, test_indexes = cls.split_data(data=indexes, probability=1 - test_share)
        return ([data_x[i] for i in train_indexes], [data_x[i] for i in test_indexes],
                [data_y[i] for i in train_indexes], [data_y[i] for i in test_indexes])

    @staticmethod
    def accuracy(true_positive: int, false_positive: int, false_negative: int,
                 true_negative: int) -> float:
        """
        How close or far off a given set of measurements are to their true value
        :param true_positive: Actual positive
        :param false_positive: Wrong positive
        :param false_negative: Wrong negative
        :param true_negative: Actual negative
        :return: Accuracy score
        """
        correct = true_positive + true_negative
        total = true_positive + false_positive + true_negative + false_negative
        return correct / total

    @staticmethod
    def precision(true_positive: int, false_positive: int) -> float:
        """
        How accurate our positive predictions are
        :param true_positive: Actual positive
        :param false_positive: Wrong positive
        :return: Precision score
        """
        return true_positive / (true_positive + false_positive)

    @staticmethod
    def recall(true_positive: int, false_negative: int) -> float:
        """
        What fraction of the positives our model identified
        :param true_positive: Actual positive
        :param false_negative: Wrong negative
        :return: Recall score
        """
        return true_positive / (true_positive + false_negative)

    @classmethod
    def f1_score(cls, true_positive: int, false_positive: int, false_negative: int) -> float:
        """
        Harmonic mean of precision and recall
        :param true_positive: Actual positive
        :param false_positive: Wrong positive
        :param false_negative: Wrong negative
        :return: F1 score
        """
        precision = cls.precision(true_positive=true_positive, false_positive=false_positive)
        recall = cls.recall(true_positive=true_positive, false_negative=false_negative)
        return 2 * precision * recall / (precision + recall)
