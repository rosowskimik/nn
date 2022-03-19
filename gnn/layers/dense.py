from dataclasses import dataclass
from typing_extensions import Self
import numpy as np

from gnn.layers.base import BaseLayer


@dataclass()
class DenseLayer(BaseLayer):
    """A layer, where each input is connected to all outputs."""

    weights: np.ndarray
    biases: np.ndarray

    def __init__(self, input_size: int, output_size: int):
        """Create new `DenseLayer` with random weights and biases which maps `input_size` inputs to `output_size` outputs."""
        self.weights = (np.random.randn(output_size, input_size) - 0.5) * 2.0
        self.biases = (np.random.randn(output_size, 1) - 0.5) * 2.0

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.dot(self.weights, self.input) + self.biases

    @staticmethod()
    def cross_layer(l1: Self, l2: Self) -> Self:
        """Creates new layer from crossing of two layers."""
        assert l1.weights.shape == l2.weights.shape
        assert l1.biases.shape == l2.biases.shape

        output_size, input_size = l1.weights.shape
        crossed_layer = DenseLayer(input_size, output_size)

        for idx in np.ndindex(crossed_layer.weights):
            crossed_layer.weights[idx] = (
                l1.weights[idx] if np.random.random() < 0.5 else l2.weights[idx]
            )

        for idx in np.ndindex(crossed_layer.biases):
            crossed_layer.biases[idx] = (
                l1.biases[idx] if np.random.random() < 0.5 else l2.biases[idx]
            )

    def mutate_layer(self, mutation_rate: float):
        """Randomly changes layer's weights and biases, based on `mutation_rate`."""
        for idx in np.ndindex(self.weights):
            if np.random.random() <= mutation_rate:
                self.weights[idx] = (np.random.random() - 0.5) * 2.0

        for idx in np.ndindex(self.biases):
            if np.random.random() <= mutation_rate:
                self.biases[idx] = (np.random.random() - 0.5) * 2.0
