"""
Testing k nearest neighbors implementation
"""

import random
from collections import Counter, defaultdict
from typing import Dict, Tuple
from unittest import TestCase

from ..k_nearest_neighbors import KNearestNeighbors as knn


class TestKNearestNeighbors(TestCase):
    """
    This class contains tests for k-nearest neighbors implementation
    """

    labels = ['a', 'b', 'c', 'b']
    labels_tie = ['a', 'b', 'c', 'b', 'a']

    def test_raw_majority_vote(self):
        """Successfully test"""
        self.assertEqual(knn.raw_majority_vote(labels=self.labels), 'b')

    def test_majority_vote(self):
        """Successfully test"""
        self.assertEqual(knn.majority_vote(labels=self.labels_tie), 'b')

    def test_load_iris_data(self):
        """Successfully test"""
        random.seed(17)
        train, test = knn.load_iris_data()
        counter = Counter([x.label for x in train])
        self.assertEqual(len(train), 105)
        self.assertEqual(len(test), 45)
        self.assertEqual(counter['0'], 38)
        self.assertEqual(counter['1'], 34)
        self.assertEqual(counter['2'], 33)

    def test_knn_classify(self):
        """Successfully test"""
        random.seed(17)
        confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
        num_correct = 0
        train, test = knn.load_iris_data()
        for iris in test:
            predicted = knn.knn_classify(num_neighbors=5, labeled_points=train,
                                         new_point=iris.point)
            actual = iris.label
            if predicted == actual:
                num_correct += 1
            confusion_matrix[(predicted, actual)] += 1
        share_correct = num_correct / len(test)
        self.assertTrue(0.954 <= share_correct <= 0.956)
        self.assertEqual(confusion_matrix[('0', '0')], 12)
        self.assertEqual(confusion_matrix[('1', '1')], 15)
        self.assertEqual(confusion_matrix[('1', '2')], 1)
        self.assertEqual(confusion_matrix[('2', '1')], 1)
        self.assertEqual(confusion_matrix[('2', '2')], 16)
