"""
Tests for Naive Bayes implementation
"""

from unittest import TestCase

from ..naive_bayes import Message as m
from ..naive_bayes import NaiveBayes as nb


class TestNaiveBayes(TestCase):
    """
    Tests for Naive Bayes implementation
    """

    messages = [
        m(text="spam rules", is_spam=True),
        m(text="ham rules", is_spam=False),
        m(text="hello ham", is_spam=False),
    ]

    def test_tokenize(self):
        """Successfully test"""
        words = {"data", "science", "is"}
        self.assertEqual(nb.tokenize(text="Data Science is science"), words)

    def test_train(self):
        """Successfully test"""
        model = nb()
        model.train(messages=self.messages)
        self.assertEqual(model.tokens, {"spam", "ham", "rules", "hello"})
        self.assertEqual(model.spam_messages, 1)
        self.assertEqual(model.ham_messages, 2)
        self.assertEqual(model.token_spam_counts, {"spam": 1, "rules": 1})
        self.assertEqual(model.token_ham_counts, {"ham": 2, "rules": 1, "hello": 1})

    def test_predict(self):
        """Successfully test"""
        model = nb()
        model.train(messages=self.messages)
        probability = model.predict(text="hello spam")
        self.assertTrue(0.82 <= probability <= 0.84)
