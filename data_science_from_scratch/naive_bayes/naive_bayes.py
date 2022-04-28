"""
Naive Bayes implementation
"""

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, NamedTuple, NoReturn, Set, Tuple


class Message(NamedTuple):
    """
    Data class for Naive Bayes Classifier
    """
    text: str
    is_spam: bool


class NaiveBayes:
    """
    Naive Bayes implementation class
    """

    def __init__(self, alpha: float = 0.5) -> NoReturn:
        """
        Creates a Naive Bayes Classifier with an alpha = 0.5
            and set up all the variables needed to calculate the probabilities
        :param alpha: Smoothing factor
        """
        self.alpha: float = alpha
        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages: int = 0
        self.ham_messages: int = 0

    def train(self, messages: Iterable[Message]) -> NoReturn:
        """
        Counts how many spam and ham messages do we have
        Counts if a word appears into a spam or ham message
            How many spam/ham message does this word appear
        :param messages: Messages to train the model
        :return: NoReturn
        """
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1
            for token in self.tokenize(text=message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """
        Calculates P(token|spam) and P(token|ham)
        :param token: A word
        :return: Probability of being spam and ham depending on the word
        """
        spam: int = self.token_spam_counts[token]
        ham: int = self.token_ham_counts[token]
        p_token_spam: float = (spam + self.alpha) / (self.spam_messages + 2 * self.alpha)
        p_token_ham: float = (ham + self.alpha) / (self.ham_messages + 2 * self.alpha)
        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        """
        Tokenize a message
        Find the probability of being spam and ham for each word in the vocab
        If the token appears in the message
            We add the log of the prob
        If not
            We add the log of 1 - prob
        :param text: Input message to be classified
        :return: Probability of being a spam message
        """
        text_tokens: Set[str] = self.tokenize(text=text)
        log_prob_if_spam: float = 0.0
        log_prob_if_ham: float = 0.0
        for token in self.tokens:
            probabilities: Tuple[float, float] = self._probabilities(token=token)
            if token in text_tokens:
                log_prob_if_spam += math.log(probabilities[0])
                log_prob_if_ham += math.log(probabilities[1])
            else:
                log_prob_if_spam += math.log(1.0 - probabilities[0])
                log_prob_if_ham += math.log(1.0 - probabilities[1])
        prob_if_spam: float = math.exp(log_prob_if_spam)
        prob_if_ham: float = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)

    @staticmethod
    def tokenize(text: str) -> Set[str]:
        """
        Convert text to lower case, extract words, numbers, and apostrophes; and
            return the unique occurrences
        :param text: Input text
        :return: Set with unique words
        """
        text: str = text.lower()
        all_words: List[str] = re.findall("[a-z0-9']+", text)
        return set(all_words)
