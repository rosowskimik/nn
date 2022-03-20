from dataclasses import dataclass
from itertools import pairwise
from typing import Iterator
from typing_extensions import Self
import jsonpickle

import numpy as np
from nn.layers.activators import ActivationLayer
from nn.layers import BaseLayer, Dense


@dataclass()
class NeuralNetwork:
    layers: list[BaseLayer]

    input_count: int

    def __init__(self, layers: list[BaseLayer]) -> None:
        assert len(layers) >= 3, "At least 3 layers required"
        assert isinstance(layers[0], Dense), "First layer should be a dense layer"

        self.layers = layers
        self.input_count = layers[0].weights.shape[1]

    @staticmethod
    def from_counts(
        neuron_counts: list[int],
        activators: list[ActivationLayer],
    ) -> "NeuralNetwork":
        assert len(neuron_counts) >= 3, "At least 3 layers required"
        assert len(neuron_counts) == len(
            activators
        ), "Each layer needs an activation function"

        layers = list()

        for counts, activator in zip(pairwise(neuron_counts), activators):
            layers.append(Dense(*counts))
            layers.append(activator)

        layers.append(activator[-1])

        return NeuralNetwork(layers)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        output = np.reshape(inputs, (self.input_count, 1))

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, output_error: np.ndarray, learning_rate: float):
        input_error = output_error

        for layer in reversed(self.layers):
            input_error = layer.backward(input_error, learning_rate)

    def cross_with(self, other: Self) -> Self:
        return NeuralNetwork(
            [l1.cross_with(l2) for l1, l2 in zip(self.layers, other.layers)]
        )

    def mutate_network(self, mutation_rate: float):
        for layer in self.layers:
            layer.mutate_layer(mutation_rate)

    def to_json(self) -> str:
        return jsonpickle.encode(self)

    @staticmethod
    def from_json(json: str) -> "NeuralNetwork":
        return jsonpickle.decode(json)

    def __iter__(self) -> Iterator[BaseLayer]:
        return iter(self.layers)
