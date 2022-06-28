"""
Neural Networks implementation testing module
"""

import random
from unittest import TestCase

from ..neural_networks import NeuralNetworks as Nn


class TestNeuralNetworks(TestCase):
    """
    Neural Networks implementation testing class
    """
    random.seed(0)
    values_x = [[0., 0], [0., 1], [1., 0], [1., 1]]
    values_y = [[0.], [1.], [1.], [0.]]
    network = [
        # hidden layer: 2 inputs -> 2 outputs
        [
            [random.random() for _ in range(2 + 1)],    # 1st hidden neuron
            [random.random() for _ in range(2 + 1)]     # 2nd hidden neuron
        ],
        # output layer: 2 inputs -> 1 output
        [
            [random.random() for _ in range(2 + 1)]     # 1st output neuron
        ]
    ]
    INPUT_SHAPE = 10
    network = Nn.fit(steps=50, learning_rate=1.0, values_x=values_x, values_y=values_y, network=network)
    values_x_fizz_buzz = [Nn.binary_encode(value_x=value, size=10) for value in range(101, 1024)]
    values_y_fizz_buzz = [Nn.fizz_buzz_encode(value_x=value) for value in range(101, 1024)]
    NUM_HIDDEN = 25
    network_fizz_buzz = [
        # hidden layer: 10 inputs -> NUM_HIDDEN outputs
        [
            [random.random() for _ in range(10 + 1)] for _ in range(25)
        ],
        # output_layer: NUM_HIDDEN inputs -> 4 outputs
        [
            [random.random() for _ in range(25 + 1)] for _ in range(4)
        ]
    ]
    network_fizz_buzz = Nn.fit_with_loss(steps=50, values_x=values_x_fizz_buzz,
                                         values_y=values_y_fizz_buzz, network=network_fizz_buzz,
                                         learning_rate=1.0)

    def test_fit(self):
        self.assertTrue(Nn.feed_forward(self.network, [0, 0])[-1][0] < 0.5)

    def test_fizz_buzz(self):
        correct_size = 0
        for number in range(1, 101):
            value_x = Nn.binary_encode(value_x=number, size=self.INPUT_SHAPE)
            predicted = Nn.argmax(
                values_x=Nn.feed_forward(
                    neural_network=self.network_fizz_buzz, input_vector=value_x)[-1])
            actual = Nn.argmax(values_x=Nn.fizz_buzz_encode(value_x=number))
            if predicted == actual:
                correct_size += 1
        self.assertTrue(correct_size > 50)

    def test_step_function(self):
        self.assertEqual(Nn.step_function(value_x=2), 1.0)

    def test_perceptron_output(self):
        self.assertEqual(Nn.perceptron_output(weights=[1, 2, 3], bias=0, value_x=[1, 2, 3]), 1.0)
