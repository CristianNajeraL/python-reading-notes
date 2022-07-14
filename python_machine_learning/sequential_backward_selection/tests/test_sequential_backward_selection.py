"""
Sequential Backward Selection test module implementation
"""

from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from ..sequential_backward_selection import SequentialBackwardSelection as Sbs


class TestSequentialBackwardSelection(TestCase):
    """
    This class contains tests for Sequential Backward Selection methods
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    df_wine = pd.read_csv(url, header=None)
    df_wine.columns = [
        'Class label',
        'Alcohol',
        'Malic acid',
        'Ash',
        'Alcalinity of ash',
        'Magnesium',
        'Total phenols',
        'Flavanoids',
        'Nonflavanoid phenols',
        'Proanthocyanins',
        'Color intensity',
        'Hue',
        'OD280/OD315 of diluted wines',
        'Proline'
    ]
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17, stratify=y)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    def test_sbs(self):
        sbs = Sbs(estimator=self.knn, k_features=1)
        sbs.fit(self.X_train_std, self.y_train)
        self.assertEqual(np.argsort(sbs.scores_)[-1], 10)
