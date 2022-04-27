"""
Testing machine learning implementation
"""

from unittest import TestCase

from ..machine_learning import MachineLearning as ml


class TestMachineLearning(TestCase):
    """
    This class contains tests for machine learning implementation
    """

    data_x = list(i for i in range(1000))
    data_y = [2 * x for x in data_x]

    def test_split_data(self):
        """Successfully test"""
        train, test = ml.split_data(data=self.data_x, probability=0.75)
        self.assertEqual(len(train), 750)
        self.assertEqual(len(test), 250)

    def test_train_test_split(self):
        """Successfully test"""
        x_train, x_test, y_train, y_test = ml.train_test_split(
            data_x=self.data_x, data_y=self.data_y, test_share=0.25)
        self.assertEqual(len(x_train), 750)
        self.assertEqual(len(x_test), 250)
        self.assertTrue(len(x_train) == len(y_train))
        self.assertTrue(len(x_test) == len(y_test))

    def test_accuracy(self):
        """Successfully test"""
        accuracy = ml.accuracy(true_positive=70, false_positive=4930, false_negative=13930,
                               true_negative=981070)
        self.assertAlmostEqual(accuracy, 0.98114)

    def test_precision(self):
        """Successfully test"""
        precision = ml.precision(true_positive=70, false_positive=4930)
        self.assertAlmostEqual(precision, 0.014)

    def test_recall(self):
        """Successfully test"""
        recall = ml.recall(true_positive=70, false_negative=13930)
        self.assertAlmostEqual(recall, 0.005)

    def test_f1_score(self):
        """Successfully test"""
        f1_score = ml.f1_score(true_positive=70, false_positive=4930, false_negative=13930)
        self.assertTrue(0.007 <= f1_score <= 0.0075)
