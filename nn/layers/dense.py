from dataclasses import dataclass
from typing_extensions import Self
import numpy as np

from nn.layers.base import BaseLayer


@dataclass()
class Dense(BaseLayer):
    """A layer, where each input is connected to all outputs."""

    weights: np.ndarray
    biases: np.ndarray

    def __init__(self, input_size: int, neuron_count: int):
        """Create new `DenseLayer` with random weights and biases which maps `input_size` inputs to `neuron_count` neurons."""
        self.weights = np.random.randn(neuron_count, input_size)
        self.biases = np.random.randn(neuron_count, 1)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        super().forward(inputs)
        return np.dot(self.weights, inputs) + self.biases

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        weights_gradient = np.dot(output_gradient, self.inputs.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

    def cross_with(self, other: Self) -> Self:
        """Creates new layer from crossing of two layers."""
        assert self.weights.shape == other.weights.shape
        assert self.biases.shape == other.biases.shape

        neuron_count, input_size = self.weights.shape
        crossed_layer = Dense(input_size, neuron_count)

        for idx in np.ndindex(crossed_layer.weights.shape):
            crossed_layer.weights[idx] = (
                self.weights[idx] if np.random.random() < 0.5 else other.weights[idx]
            )

        for idx in np.ndindex(crossed_layer.biases.shape):
            crossed_layer.biases[idx] = (
                self.biases[idx] if np.random.random() < 0.5 else other.biases[idx]
            )

        return crossed_layer

    def mutate_layer(self, mutation_rate: float):
        """Randomly changes layer's weights and biases, based on `mutation_rate`."""
        for idx in np.ndindex(self.weights.shape):
            if np.random.random() <= mutation_rate:
                self.weights[idx] = np.random.random() * 2 - 1

        for idx in np.ndindex(self.biases.shape):
            if np.random.random() <= mutation_rate:
                self.biases[idx] = np.random.random() * 2 - 1
