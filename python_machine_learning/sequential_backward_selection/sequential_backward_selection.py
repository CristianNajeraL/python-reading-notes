"""
Sequential Backward Selection implementation
"""

from itertools import combinations
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SequentialBackwardSelection:  # pylint: disable=R0913 disable=R0902
    """
    Sequential Backward Selection class

    Attributes:
        indices_: tuple
            Index feature combination
        subsets_: list
            All tested indices combinations
        scores_: list
            All tested scores
        k_score_: float
            Last score
    """

    def __init__(self, estimator: Any, k_features: int, scoring: Any = accuracy_score,
                 test_size: float = 0.25, random_state: int = 17):
        """
        Class initializer
        :param estimator: Machine learning algorithm to use
        :param k_features: Target number of features
        :param scoring: Metric to rank
        :param test_size: Size of testing data set
        :param random_state: Random state
        """
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

        self.subsets_: list = []
        self.scores_: list = []
        self.indices_: tuple = tuple()
        self.k_score_: float = 0.0

    def fit(self, x_values: np.ndarray, y_values: np.ndarray) -> object:
        """
        Fit training data, it is executed until the while loop is broken
        :param x_values: nd-array, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features
        :param y_values: 1d-array, shape = [n_examples]
            Target values
        :return:
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x_values, y_values,
            test_size=self.test_size,
            random_state=self.random_state
        )
        features_dim = x_train.shape[1]
        self.indices_ = tuple(range(features_dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(
            x_train, y_train,
            x_test, y_test,
            self.indices_
        )
        self.scores_ = [score]
        while features_dim > self.k_features:
            scores = []
            subsets = []
            for subset in combinations(self.indices_, r=features_dim - 1):
                score = self._calc_score(x_train, y_train, x_test, y_test, subset)
                scores.append(score)
                subsets.append(subset)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            features_dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self

    def _calc_score(self, x_train: np.ndarray, y_train: np.ndarray,
                    x_test: np.ndarray, y_test: np.ndarray, indices: tuple) -> float:
        """

        :param x_train: nd-array, shape = [n_examples, n_features]
            Subset of x_values
        :param y_train: 1d-array, shape = [n_examples]
            Subset of y_values
        :param x_test: nd-array, shape = [n_examples, n_features]
            Subset of x_values
        :param y_test: 1d-array, shape = [n_examples]
            Subset of y_values
        :param indices: tuple
            Index of feature to be used
        :return:
        """
        self.estimator.fit(x_train[:, indices], y_train)
        y_pred = self.estimator.predict(x_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
