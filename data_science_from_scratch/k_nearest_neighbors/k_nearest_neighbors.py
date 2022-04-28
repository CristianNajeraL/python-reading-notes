"""
k-Nearest Neighbors implementation
"""

from collections import Counter
from typing import List, NamedTuple, Tuple

from sklearn.datasets import load_iris

from ..linear_algebra.vectors import Vector
from ..linear_algebra.vectors import Vectors as v
from ..machine_learning.machine_learning import MachineLearning as ml


class LabeledPoint(NamedTuple):
    """
    Data type for k-Nearest Neighbors
    """
    point: Vector
    label: str


class KNearestNeighbors:
    """
    k-Nearest Neighbors methods
    """

    @staticmethod
    def raw_majority_vote(labels: List[str]) -> str:
        """
        Find the label with more occurrence
        :param labels: Labeled vector
        :return: Most frequent label
        """
        votes = Counter(labels)
        winner, _ = votes.most_common(1)[0]
        return winner

    @classmethod
    def majority_vote(cls, labels: List[str]) -> str:
        """
        Find the label with more occurrence even if there is a
            tie in the data set.
        Assumes that labels are ordered from nearest to farthest.
        :param labels: Labeled vector
        :return: Most frequent label
        """
        vote_counts = Counter(labels)
        winner, winner_count = vote_counts.most_common(1)[0]
        num_winners = len(
            [count for count in vote_counts.values() if count == winner_count]
        )
        if num_winners == 1:
            return winner
        return cls.majority_vote(labels=labels[:-1])

    @classmethod
    def knn_classify(cls, num_neighbors: int, labeled_points: List[LabeledPoint],
                     new_point: Vector) -> str:
        """
        Classify an unseen data point based on the distance and a specific number of neighbors
        :param num_neighbors: Number of neighbors
        :param labeled_points: Labeled vector
        :param new_point: Unseen data
        :return: Classification for the unseen data
        """
        by_distance = sorted(labeled_points,
                             key=lambda lp: v.distance(vector_x=lp.point, vector_y=new_point))
        k_nearest_labels = [lp.label for lp in by_distance[:num_neighbors]]
        return cls.majority_vote(labels=k_nearest_labels)

    @staticmethod
    def load_iris_data(share: float = 0.7) -> Tuple[List[LabeledPoint], List[LabeledPoint]]:
        """
        Extract Iris data from sklearn and transforms it into the required format
        :return: Dict with data set
        """
        data = load_iris()
        data = [
            LabeledPoint(point=[float(i) for i in x], label=str(y))
            for x, y in zip(data['data'], data['target'])
        ]
        iris_train, iris_test = ml.split_data(data=data, probability=share)
        return iris_train, iris_test
