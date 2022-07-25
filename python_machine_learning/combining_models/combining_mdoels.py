"""
Ensemble, bagging and boosting implementation
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              VotingClassifier)
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


class CombiningModels:  # pylint: disable=R0914 disable=R0902
    """
    Using sklearn to combine models
    """

    def __init__(self):
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
        self.df_wine = pd.read_csv(self.url, header=None)
        self.df_wine.columns = [
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
        self.df_wine = self.df_wine[self.df_wine['Class label'] != 1]
        self.x_values = self.df_wine[["Alcohol", "OD280/OD315 of diluted wines"]].values
        self.y_values = self.df_wine["Class label"].values
        self.label_encoder = LabelEncoder()
        self.y_values = self.label_encoder.fit_transform(self.y_values)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_values, self.y_values,
            test_size=0.2, random_state=17, stratify=self.y_values)
        self.tree = DecisionTreeClassifier(
            criterion="entropy",
            random_state=17
        )
        self.tree = self.tree.fit(self.x_train, self.y_train)
        self.y_train_pred = self.tree.predict(self.x_train)
        self.y_test_pred = self.tree.predict(self.x_test)
        self.tree_train = f1_score(self.y_train, self.y_train_pred)
        self.tree_test = f1_score(self.y_test, self.y_test_pred)

    @staticmethod
    def ensemble_pipeline():
        """
        Execute a ensemble pipeline
        :return:
        """
        iris = datasets.load_iris()
        x_values, y_values = iris['data'][:, [1, 2]], iris['target']
        label_encoder = LabelEncoder()
        y_values = label_encoder.fit_transform(y_values)
        x_train, x_test, y_train, y_test = train_test_split(x_values, y_values,
                                                            test_size=0.5, stratify=y_values,
                                                            random_state=17)
        classifier_1 = LogisticRegression(random_state=17)
        classifier_2 = DecisionTreeClassifier(random_state=17)
        classifier_3 = KNeighborsClassifier()
        scorer = make_scorer(f1_score, pos_label=1, average='micro')
        feature_selector_1 = SequentialFeatureSelector(classifier_1, scoring=scorer, n_jobs=-1)
        feature_selector_2 = SequentialFeatureSelector(classifier_2, scoring=scorer, n_jobs=-1)
        feature_selector_3 = SequentialFeatureSelector(classifier_3, scoring=scorer, n_jobs=-1)
        classifier_1_pipe = Pipeline(
            [('scaler1', StandardScaler()), ('sfs1', feature_selector_1), ('logreg', classifier_1)])
        classifier_2_pipe = Pipeline(
            [('scaler1', StandardScaler()), ('sfs2', feature_selector_2), ('dt', classifier_2)])
        classifier_3_pipe = Pipeline(
            [('scaler1', StandardScaler()), ('sfs3', feature_selector_3), ('knn', classifier_3)])
        voting_classifier = VotingClassifier(
            estimators=[
                ('pipe1', classifier_1_pipe),
                ('pipe2', classifier_2_pipe), ('pipe3', classifier_3_pipe)],
            n_jobs=-1,
            verbose=0
        )
        params = {
            'pipe1__sfs1__direction': ['forward', 'backward'],
            'pipe2__sfs2__direction': ['forward', 'backward'],
            'pipe3__sfs3__direction': ['forward', 'backward'],
            'pipe1__sfs1__cv': [2],
            'pipe2__sfs2__cv': [2],
            'pipe3__sfs3__cv': [2],

            'pipe1__sfs1__estimator__C': [10 ** x for x in np.arange(-5, 3, 0.1)],
            'pipe1__sfs1__estimator__max_iter': list(np.arange(100, 1001, 100)),

            'pipe1__logreg__C': [10 ** x for x in np.arange(-5, 3, 0.1)],
            'pipe1__logreg__max_iter': list(np.arange(100, 1001, 100)),

            'pipe2__dt__max_depth': list(np.arange(2, 11, 1)),
            'pipe2__dt__max_leaf_nodes': [2, 3, 4, 5],

            'pipe2__sfs2__estimator__max_depth': list(np.arange(2, 11, 1)),
            'pipe2__sfs2__estimator__max_leaf_nodes': [2, 3, 4, 5],
        }
        grid = RandomizedSearchCV(
            voting_classifier,
            params,
            scoring=scorer,
            cv=5,
            n_iter=10,
            n_jobs=-1
        )
        grid = grid.fit(x_train, y_train)
        return grid.score(x_test, y_test)

    def bagging(self):
        """

        :return:
        """
        bag_tree = DecisionTreeClassifier(
            criterion="entropy",
            random_state=17,
            max_depth=5
        )
        bag = BaggingClassifier(
            base_estimator=bag_tree,
            n_estimators=500,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            n_jobs=-1,
            random_state=1
        )
        bag = bag.fit(self.x_train, self.y_train)
        y_train_pred = bag.predict(self.x_train)
        y_test_pred = bag.predict(self.x_test)
        bag_train = f1_score(self.y_train, y_train_pred)
        bag_test = f1_score(self.y_test, y_test_pred)
        return bag_train, bag_test

    def boosting(self):
        """

        :return:
        """
        boost_tree = DecisionTreeClassifier(
            criterion="entropy",
            random_state=17,
            max_depth=5
        )
        ada = AdaBoostClassifier(
            base_estimator=boost_tree,
            n_estimators=500,
            learning_rate=0.01,
            random_state=17
        )
        ada.fit(self.x_train, self.y_train)
        y_train_pred = ada.predict(self.x_train)
        y_test_pred = ada.predict(self.x_test)
        ada_train = f1_score(self.y_train, y_train_pred)
        ada_test = f1_score(self.y_test, y_test_pred)
        return ada_train, ada_test
