"""
Sklearn Classifiers test module implementation
"""

from unittest import TestCase

import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ..sklearn_classifiers import SklearnClassifiers as SkClass


class TestSklearnClassifiers(TestCase):
    """
    This class contains tests for Sklearn Classifier methods
    """
    RANDOM_SATE = 17
    SOLVER = "lbfgs"
    classifiers = [
        Perceptron(eta0=0.001, random_state=RANDOM_SATE),
        LogisticRegression(C=100.0, random_state=RANDOM_SATE, solver=SOLVER, multi_class="ovr"),
        LogisticRegression(C=100.0, random_state=RANDOM_SATE, solver=SOLVER, multi_class="multinomial"),
        LogisticRegression(C=1.0, random_state=RANDOM_SATE, solver=SOLVER, multi_class="ovr"),
        LogisticRegression(C=0.1, random_state=RANDOM_SATE, solver=SOLVER, multi_class="multinomial"),
        SGDClassifier(loss="log_loss", random_state=RANDOM_SATE),
        SVC(kernel="rbf", C=1.0, random_state=RANDOM_SATE, gamma=100),
        DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=RANDOM_SATE),
        RandomForestClassifier(criterion="gini", n_estimators=25, random_state=RANDOM_SATE),
        KNeighborsClassifier(n_neighbors=4, p=2, metric="minkowski")
    ]
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SATE, stratify=y
    )
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    def test_sklearn_classifiers(self):
        """
        Testing 10 Sklearn Classifiers
        :return:
        """
        accuracy = []
        for classifier in self.classifiers:
            model = SkClass(estimator=classifier)
            model.fit(self.X_train_std, self.y_train)
            accuracy.append(model.score(self.X_test_std, self.y_test))
        self.assertTrue(all(np.array(accuracy) > 0.79))
