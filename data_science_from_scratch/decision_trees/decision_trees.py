"""
Decision trees methods
"""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, NamedTuple, TypeVar, Union

from ..linear_algebra.vectors import Value, Vector

T = TypeVar("T")


class Leaf(NamedTuple):
    """
    Leaf node data class
    """
    value: Any


class Split(NamedTuple):
    """
    Split data class
    """
    attribute: str
    subtrees: dict
    default_value: Any = None


DecisionTree = Union[Leaf, Split]


class DecisionTrees:
    """
    Decision trees methods implementation
    """

    @staticmethod
    def entropy(class_probabilities: Vector) -> Value:
        """
        Given a list of class probabilities, compute the entropy
        :param class_probabilities: Vector of probabilities
        :return: Entropy value
        """
        return sum(
            -probability * math.log(
                probability, 2
            ) for probability in class_probabilities if probability > 0
        )

    @staticmethod
    def class_probabilities(labels: List[Any]) -> Vector:
        """
        Compute the probability of a given vector
        :param labels: Vector with classes
        :return: Vector with probabilities by vector
        """
        total_count = len(labels)
        return [
            count / total_count for count in Counter(labels).values()
        ]

    @classmethod
    def data_entropy(cls, labels: List[Any]) -> Value:
        """
        Compute the entropy of a given vector of classes
        :param labels: Vector with classes
        :return: Entropy value
        """
        return cls.entropy(class_probabilities=cls.class_probabilities(labels=labels))

    @classmethod
    def partition_entropy(cls, subsets: List[List[Any]]) -> Value:
        """
        Returns the entropy from this partition of data into subsets
        :param subsets: Subset of data points
        :return: Entropy value
        """
        total_count = sum(len(subset) for subset in subsets)
        return sum(
            (
                cls.data_entropy(labels=subset) * len(subset)
            ) / total_count for subset in subsets
        )

    @staticmethod
    def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
        """
        Partition the inputs into lists based on the specified attribute
        :param inputs: Vector with values
        :param attribute:
        :return: Dictionary with partitions
        """
        partitions: Dict[Any, List[T]] = defaultdict(list)
        for input_ in inputs:
            key = getattr(input_, attribute)
            partitions[key].append(input_)
        return partitions

    @classmethod
    def partition_entropy_by(cls, inputs: List[Any], attribute: str, label_attribute: str) -> Value:
        """
        Compute the entropy corresponding to the given partition
        :param inputs: Vector with values
        :param attribute:
        :param label_attribute:
        :return: Entropy of a partition
        """
        partitions = cls.partition_by(inputs=inputs, attribute=attribute)
        labels = [
            [
                getattr(input_, label_attribute) for input_ in partition
            ] for partition in partitions.values()
        ]
        return cls.partition_entropy(subsets=labels)

    @classmethod
    def classify(cls, tree: DecisionTree, input_: Any) -> Any:
        """
        Classify the input using the given decision tree
        :param tree: Decision tree
        :param input_: Input vector
        :return: Predicted value
        """
        if isinstance(tree, Leaf):
            return tree.value
        subtree_key = getattr(input_, tree.attribute)
        if subtree_key not in tree.subtrees:
            return tree.default_value
        subtree = tree.subtrees[subtree_key]
        return cls.classify(tree=subtree, input_=input_)

    @classmethod
    def build_tree_id3(cls,
                       inputs: List[Any],
                       split_attributes: List[str],
                       target_attribute: str) -> DecisionTree:
        """
        Generates a decision tree
        :param inputs: Inputs value
        :param split_attributes:
        :param target_attribute:
        :return: A decision tree
        """
        label_counts = Counter(getattr(input_, target_attribute) for input_ in inputs)
        most_common_label = label_counts.most_common(1)[0][0]
        if len(label_counts) == 1:
            return Leaf(most_common_label)
        if not split_attributes:
            return Leaf(most_common_label)

        def split_entropy(attribute: str) -> Value:
            """
            Helper function for finding the best attribute
            :param attribute:
            :return: Best attribute to split on
            """
            return cls.partition_entropy_by(
                inputs=inputs, attribute=attribute, label_attribute=target_attribute)

        best_attribute = min(split_attributes, key=split_entropy)
        partitions = cls.partition_by(inputs=inputs, attribute=best_attribute)
        new_attributes = [
            attribute_ for attribute_ in split_attributes if attribute_ != best_attribute]
        subtrees = {
            attribute_value: cls.build_tree_id3(
                inputs=subset,
                split_attributes=new_attributes,
                target_attribute=target_attribute
            ) for attribute_value, subset in partitions.items()
        }
        return Split(attribute=best_attribute, subtrees=subtrees, default_value=most_common_label)
