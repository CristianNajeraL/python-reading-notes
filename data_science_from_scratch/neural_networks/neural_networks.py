"""
Neural Networks implementation module
"""

import math
from typing import List

import tqdm

from ..gradient_descent.gradient_descent import GradientDescent as Gd
from ..linear_algebra.vectors import Value, Vector, Vectors


class NeuralNetworks:  # pylint: disable=R0913 disable=unbalanced-tuple-unpacking
    """
    Neural Networks implementation class
    """

    @staticmethod
    def step_function(value_x: Value) -> Value:
        """
        Binary function
        :param value_x: Random value
        :return: 1 if value_x is bigger than 0, 0 otherwise
        """
        return 1.0 if value_x >= 0 else 0.0

    @classmethod
    def perceptron_output(cls, weights: Vector, bias: Value, value_x: Vector) -> Value:
        """
        Returns 1 if the perceptron 'fires', 0 if not
        :param weights: Vector with NN weights
        :param bias: Bias value
        :param value_x: Vector with random numbers
        :return: 1 or 0 from step function
        """
        calculation = Vectors.dot(vector_x=weights, vector_y=value_x) + bias
        return cls.step_function(value_x=calculation)

    @staticmethod
    def sigmoid(value_x: Value) -> Value:
        """
        Sigmoid calculation
        :param value_x: Random value
        :return: Value after sigmoid activation
        """
        return 1 / (1 + math.exp(-value_x))

    @classmethod
    def neuron_output(cls, weights: Vector, inputs: Vector) -> Value:
        """
        Weights includes the bias term, inputs includes a 1
        Neuron after sigmoid activation
        :param weights: Vector with NN weights
        :param inputs: Vector with random numbers
        :return: Value after sigmoid activation
        """
        return cls.sigmoid(value_x=Vectors.dot(vector_x=weights, vector_y=inputs))

    @classmethod
    def feed_forward(cls, neural_network: List[List[Vector]], input_vector: Vector) -> List[Vector]:
        """
        Feeds the input vector through the neural network.
        Returns the outputs of all layers (not just the last one).
        :param neural_network: NN structure (hidden and output layers)
        :param input_vector: Vector with random numbers
        :return: The outputs of all the layers
        """
        outputs: List[Vector] = []
        for layer in neural_network:
            input_with_bias = input_vector + [1]
            output = [
                cls.neuron_output(weights=neuron, inputs=input_with_bias) for neuron in layer
            ]
            outputs.append(output)
            input_vector = output
        return outputs

    @classmethod
    def squared_error_gradients(cls, network: List[List[Vector]], input_vector: Vector,
                                target_vector: Vector) -> List[List[Vector]]:
        """
        Given a neural network, an input vector, and a target vector,
        make a prediction and compute the gradient of the squared error
        loss with respect to the neuron weights.
        :param network: NN structure
        :param input_vector: Vector with random numbers
        :param target_vector: Vector with random numbers or target
        :return: NN structure gradients
        """
        hidden_outputs, outputs = cls.feed_forward(
            neural_network=network, input_vector=input_vector)
        output_deltas = [
            output * (1 - output) * (output - target) for output, target in zip(
                outputs, target_vector)
        ]
        output_grads = [
            [
                output_deltas[i] * hidden_output for hidden_output in hidden_outputs + [1]
            ] for i, output_neuron in enumerate(network[-1])
        ]
        hidden_deltas = [
            hidden_output * (1 - hidden_output) * Vectors.dot(
                vector_x=output_deltas, vector_y=[
                    n[i] for n in network[-1]
                ]
            ) for i, hidden_output in enumerate(hidden_outputs)
        ]
        hidden_grads = [
            [
                hidden_deltas[i] * input_ for input_ in input_vector + [1]
            ] for i, hidden_neuron in enumerate(network[0])
        ]
        return [hidden_grads, output_grads]

    @classmethod
    def fit(cls, steps: Value, learning_rate: Value, values_x: List[Vector], values_y: List[Vector],
            network: List[List[Vector]]) -> List[List[Vector]]:
        """
        Fit NN with gradient descendent
        :param steps: Number of epochs
        :param learning_rate: Step size
        :param values_x: Vector with random numbers
        :param values_y: Vector with random numbers or target
        :param network: NN structure
        :return: Updated NN structure
        """
        for _ in tqdm.trange(steps, desc="NN fit"):
            for value_x, value_y in zip(values_x, values_y):
                gradients = cls.squared_error_gradients(
                    network=network, input_vector=value_x, target_vector=value_y)
                network = [
                    [
                        Gd.gradient_step(vector=neuron, gradient=grad,
                                         step_size=-learning_rate) for neuron, grad in zip(
                            layer, layer_grad)
                    ] for layer, layer_grad in zip(network, gradients)
                ]
        return network

    @staticmethod
    def fizz_buzz_encode(value_x: Value) -> Vector:
        """
        Encoding for Fizz Buzz g,e
        Print the numbers 1 to 100, except that if the number is divisible
        by 3, print "fizz"; if the number is divisible by 5, print "buzz";
        and if the number is divisible by 15, print "fizzbuzz".
        :param value_x: Random value
        :return: Fizz Buzz vector
        """
        if value_x % 15 == 0:
            return [0, 0, 0, 1]
        if value_x % 5 == 0:
            return [0, 0, 1, 0]
        if value_x % 3 == 0:
            return [0, 1, 0, 0]
        return [1, 0, 0, 0]

    @staticmethod
    def binary_encode(value_x: Value, size: Value) -> Vector:
        """
        From integer to binary
        :param size: Output size
        :param value_x: Random integer
        :return: Binary representation of an integer with a defined size
        """
        binary = [int(i) for i in f"{value_x:b}"]
        actual_size = len(binary)
        if actual_size == size:
            return binary
        if actual_size < size:
            for _ in range(size - actual_size):
                binary.append(0)
        return binary[-size:]

    @classmethod
    def fit_with_loss(cls, steps: Value, values_x: List[Vector], values_y: List[Vector],
                      network: List[List[Vector]], learning_rate: Value) -> List[List[Vector]]:
        """
        Fit NN with gradient descendent
        :param steps: Number of epochs
        :param learning_rate: Step size
        :param values_x: Vector with random numbers
        :param values_y: Vector with random numbers or target
        :param network: NN structure
        :return: Updated NN structure
        """
        with tqdm.trange(steps) as step:
            for _ in step:
                epoch_loss = 0.0
                for value_x, value_y in zip(values_x, values_y):
                    predicted = cls.feed_forward(neural_network=network, input_vector=value_x)[-1]
                    epoch_loss += Vectors.squared_distance(vector_x=predicted, vector_y=value_y)
                    gradients = cls.squared_error_gradients(network=network, input_vector=value_x,
                                                            target_vector=value_y)
                    network = [
                        [
                            Gd.gradient_step(vector=neuron,
                                             gradient=grad,
                                             step_size=-learning_rate) for neuron, grad in zip(
                                layer, layer_grad)
                        ] for layer, layer_grad in zip(network, gradients)
                    ]
                step.set_description(f"Fizz Buzz (loss: {epoch_loss:.2f})")
        return network

    @staticmethod
    def argmax(values_x: List) -> Value:
        """
        Returns the index of the largest value
        :param values_x: Vector
        :return: Index of the largest value
        """
        return max(range(len(values_x)), key=lambda index: values_x[index])
