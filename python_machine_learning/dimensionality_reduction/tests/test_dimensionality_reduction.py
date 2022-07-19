"""
Dimensionality reduction test module implementation
"""

import os
from unittest import TestCase

import pandas as pd
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..dimensionality_reduction import DimensionalityReduction as Dr

path = os.path.join("https://archive.ics.uci.edu", "ml", "machine-learning-databases", "iris", "iris.data")


class TestDimensionalityReduction(TestCase):
    """
    This class contains tests for Dimensionality reduction methods
    """

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    df_wine = pd.read_csv(url, header=None)
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17, stratify=y)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    X_moons, y_moons = make_moons(n_samples=500, random_state=17)
    X_circle, y_circle = make_circles(n_samples=500, random_state=17)

    def test_pca_decomposition(self):
        lr, score = Dr(variance=0.75).pca_decomposition(
            x_train_values=self.X_train_std, x_test_values=self.X_test_std,
            y_train_values=self.y_train, y_test_values=self.y_test
        )
        self.assertAlmostEqual(score, 1.0)

    def test_lda_decomposition(self):
        lr, score = Dr(variance=0.75).lda_decomposition(
            x_train_values=self.X_train_std, x_test_values=self.X_test_std,
            y_train_values=self.y_train, y_test_values=self.y_test
        )
        self.assertAlmostEqual(score, 1.0)

    def test_kernel_pca_moons(self):
        x_train, x_test, y_train, y_test = train_test_split(self.X_moons, self.y_moons,
                                                            test_size=0.3, random_state=17,
                                                            stratify=self.y_moons)
        lr, score = Dr(variance=0.75).kernel_pca(
            x_train_values=x_train, x_test_values=x_test,
            y_train_values=y_train, y_test_values=y_test,
            gamma=2
        )
        self.assertAlmostEqual(score, 1.0)

    def test_kernel_pca_circle(self):
        x_train, x_test, y_train, y_test = train_test_split(self.X_circle, self.y_circle,
                                                            test_size=0.3, random_state=17,
                                                            stratify=self.y_moons)
        lr, score = Dr(variance=0.75).kernel_pca(
            x_train_values=x_train, x_test_values=x_test,
            y_train_values=y_train, y_test_values=y_test,
            gamma=0.2
        )
        self.assertAlmostEqual(score, 1.0)
