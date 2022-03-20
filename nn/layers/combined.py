from dataclasses import dataclass
from typing import overload
from typing_extensions import Self

import numpy as np
from nn.layers.activators import ActivationLayer
from nn.layers.base import BaseLayer
from nn.layers.dense import DenseLayer
from nn.layers.softmax import SoftmaxLayer

# return CombinedLayer(DenseLayer(input_count, output_count),activation,use_softmax)


@dataclass()
class CombinedLayer(BaseLayer):
    """Layer, which combines dense, activation and optionally softmax layers."""

    dense: DenseLayer
    activation: ActivationLayer

    use_softmax: bool = False

    @staticmethod
    def from_counts(
        input_count: int,
        neuron_count: int,
        activation: ActivationLayer,
        use_softmax: bool = False,
    ) -> "CombinedLayer":
        return CombinedLayer(
            DenseLayer(input_count, neuron_count), activation, use_softmax
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        output = self.dense.forward(inputs)
        output = self.activation.forward(output)

        if self.use_softmax:
            output = SoftmaxLayer().forward(output)

        return output

    def cross_with(self, other: Self) -> Self:
        return CombinedLayer(
            self.dense.cross_with(other.dense), self.activation, self.use_softmax
        )

    def mutate_layer(self, mutation_rate: float):
        self.dense.mutate_layer(mutation_rate)
