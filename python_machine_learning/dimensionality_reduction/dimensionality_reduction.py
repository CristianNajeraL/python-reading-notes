"""
Dimensionality reduction implementation
"""

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Lda
from sklearn.linear_model import LogisticRegression


class DimensionalityReduction:
    """
    Using dimensionality reduction to fit logistic regression with linear and non-linear data

    Attributes:
        variance: float
            Explained variance for choosing the optimal number of components
    """
    def __init__(self, variance: float):
        self.variance = variance

    def pca_decomposition(self, x_train_values: np.ndarray, x_test_values: np.ndarray,
                          y_train_values: np.ndarray, y_test_values: np.ndarray):
        """
        Fit logistic regression with PCA as inputs
        :param x_train_values:
        :param x_test_values:
        :param y_train_values:
        :param y_test_values:
        :return:
        """
        def optimal_components_size(x_values: np.ndarray, variance: float = self.variance):
            """
            Find the optimal number of components
            :param x_values:
            :param variance:
            :return:
            """
            _pca = PCA(n_components=None)
            _pca.fit(x_values)
            return np.argmax(np.cumsum(_pca.explained_variance_ratio_) >= variance) + 1

        def fit(number_of_components: int, x_train_values_: np.ndarray,
                x_test_values_: np.ndarray, y_train_values_: np.ndarray,
                y_test_values_: np.ndarray):
            """

            :param number_of_components:
            :param x_train_values_:
            :param x_test_values_:
            :param y_train_values_:
            :param y_test_values_:
            :return:
            """
            pca = PCA(n_components=number_of_components)
            logit = LogisticRegression(multi_class='ovr', random_state=17, solver='lbfgs')
            x_train_values_ = pca.fit_transform(x_train_values_)
            x_test_values_ = pca.transform(x_test_values_)
            logit.fit(x_train_values_, y_train_values_)
            return logit, logit.score(x_test_values_, y_test_values_)

        n_components = optimal_components_size(x_values=x_train_values, variance=self.variance)
        return fit(number_of_components=n_components, x_train_values_=x_train_values,
                   x_test_values_=x_test_values, y_train_values_=y_train_values,
                   y_test_values_=y_test_values)

    def lda_decomposition(self, x_train_values: np.ndarray, x_test_values: np.ndarray,
                          y_train_values: np.ndarray, y_test_values: np.ndarray):
        """
        Fit logistic regression with LDA as inputs
        :param x_train_values:
        :param x_test_values:
        :param y_train_values:
        :param y_test_values:
        :return:
        """
        def optimal_components_size(x_values: np.ndarray, y_values: np.ndarray,
                                    variance: float = self.variance):
            """
            Find the optimal number of components
            :param x_values:
            :param y_values:
            :param variance:
            :return:
            """
            _lda = Lda(n_components=None)
            _lda.fit(x_values, y_values)
            return np.argmax(np.cumsum(_lda.explained_variance_ratio_) >= variance) + 1

        def fit(number_of_components: int, x_train_values_: np.ndarray,
                x_test_values_: np.ndarray, y_train_values_: np.ndarray,
                y_test_values_: np.ndarray):
            """

            :param number_of_components:
            :param x_train_values_:
            :param x_test_values_:
            :param y_train_values_:
            :param y_test_values_:
            :return:
            """
            lda = Lda(n_components=number_of_components)
            logit = LogisticRegression(multi_class='ovr', random_state=17, solver='lbfgs')
            x_train_values_ = lda.fit_transform(x_train_values_, y_train_values_)
            x_test_values_ = lda.transform(x_test_values_)
            logit.fit(x_train_values_, y_train_values_)
            return logit, logit.score(x_test_values_, y_test_values_)

        n_components = optimal_components_size(x_values=x_train_values, y_values=y_train_values,
                                               variance=self.variance)
        return fit(number_of_components=n_components, x_train_values_=x_train_values,
                   x_test_values_=x_test_values, y_train_values_=y_train_values,
                   y_test_values_=y_test_values)

    @staticmethod
    def kernel_pca(x_train_values: np.ndarray, x_test_values: np.ndarray,
                   y_train_values: np.ndarray, y_test_values: np.ndarray, gamma: float or int):
        """
        Fit logistic regression with k-PCA as inputs
        :param y_test_values:
        :param y_train_values:
        :param x_test_values:
        :param x_train_values:
        :param gamma:
        :return:
        """
        kernel_pca_ = KernelPCA(kernel="rbf", gamma=gamma, n_jobs=-1, random_state=17)
        x_train_values_ = kernel_pca_.fit_transform(x_train_values)
        x_test_values_ = kernel_pca_.transform(x_test_values)
        logit = LogisticRegression(multi_class='ovr', random_state=17, solver='lbfgs')
        logit.fit(x_train_values_, y_train_values)
        return logit, logit.score(x_test_values_, y_test_values)
