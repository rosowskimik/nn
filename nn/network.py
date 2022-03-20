from dataclasses import dataclass
from itertools import pairwise
from typing_extensions import Self
import jsonpickle

import numpy as np
from nn.layers.activators import ActivationLayer
from nn.layers import CombinedLayer, BaseLayer


@dataclass()
class GeneticNeuralNetwork:
    layers: list[BaseLayer]

    @staticmethod
    def from_counts(
        neuron_counts: list[int],
        activators: list[ActivationLayer],
        use_softmax: bool = False,
    ) -> "GeneticNeuralNetwork":
        assert len(neuron_counts) >= 3, "At least 3 layers required"
        assert len(neuron_counts) == len(
            activators
        ), "Each layer needs an activation function"

        neuron_counts.append(neuron_counts[-1])

        layers = list()

        for counts, activator in zip(pairwise(neuron_counts), activators):
            layers.append(CombinedLayer.from_counts(*counts, activator, use_softmax))

        return GeneticNeuralNetwork(layers)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        output = inputs

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def cross_with(self, other: Self) -> Self:
        return GeneticNeuralNetwork(
            [l1.cross_with(l2) for l1, l2 in zip(self.layers, other.layers)]
        )

    def mutate_network(self, mutation_rate: float):
        for layer in self.layers:
            layer.mutate_layer(mutation_rate)

    def to_json(self) -> str:
        return jsonpickle.encode(self)

    @staticmethod
    def from_json(json: str) -> "GeneticNeuralNetwork":
        return jsonpickle.decode(json)
