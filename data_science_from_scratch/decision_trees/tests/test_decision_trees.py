"""
Testing Decision Trees methods
"""

import random
from typing import NamedTuple, Optional
from unittest import TestCase

from ..decision_trees import DecisionTrees as Dt
from ..decision_trees import Leaf, Split


class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None


class TestDecisionTrees(TestCase):
    """
    This class contains tests for Decision Trees methods
    """
    random.seed(17)
    inputs_ = [
        Candidate(
            random.choice(["Senior", "Mid", "Junior"]),
            random.choice(["Java", "R", "Python"]),
            random.choice([True, False]),
            random.choice([True, False]),
            random.choice([True, False])
        ) for _ in range(500)
    ]
    tree = Dt.build_tree_id3(
        inputs=inputs_, split_attributes=['level', 'lang', 'tweets', 'phd'], target_attribute='did_well')

    def test_partition_entropy_by(self):
        self.assertTrue(0.8 < Dt.partition_entropy_by(self.inputs_, 'level', 'did_well') < 1)
        self.assertTrue(0.8 < Dt.partition_entropy_by(self.inputs_, 'lang', 'did_well') < 1)
        self.assertTrue(0.8 < Dt.partition_entropy_by(self.inputs_, 'tweets', 'did_well') < 1)
        self.assertTrue(0.8 < Dt.partition_entropy_by(self.inputs_, 'phd', 'did_well') < 1)

    def test_classify(self):
        self.assertTrue(Dt.classify(self.tree, Candidate("Junior", "Java", True, False)))
        self.assertFalse(Dt.classify(self.tree, Candidate("Junior", "Java", True, True)))

    def test_classify_116(self):
        tree_116 = Split(
            "level",
            {
                False: Leaf(True),
                True: Leaf(False)
            },
            True
        )
        self.assertTrue(Dt.classify(tree_116, Candidate("Junior", "Java", True, False)))

    def test_build_tree_id3_132(self):
        inputs_ = [Candidate('Senior', 'Java', False, False, False)]
        tree = Dt.build_tree_id3(
            inputs=inputs_, split_attributes=['level', 'lang', 'tweets', 'phd'], target_attribute='did_well')
        self.assertTrue(isinstance(tree, Leaf))
