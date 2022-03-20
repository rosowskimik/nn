from dataclasses import dataclass
import numpy as np
from nn.layers.base import BaseLayer


@dataclass()
class ActivationLayer(BaseLayer):
    """Base activation layer class. Should not be instantiated. All derived classes should implement `activate` staticmethod."""

    def forward(self, vals: np.ndarray) -> np.ndarray:
        vectorized = np.vectorize(self.activate)

        return vectorized(vals)

    def activate(self, val: float) -> float:
        raise NotImplementedError


@dataclass()
class BinaryStep(ActivationLayer):
    threshhold: float = 0.0

    def activate(self, val: float) -> float:
        return 0.0 if val < self.threshhold else 1.0


@dataclass()
class Linear(ActivationLayer):
    def activate(self, val: float) -> float:
        return val


@dataclass()
class Logistic(ActivationLayer):
    def activate(self, val: float) -> float:
        return 1 / (1 + np.exp(-val))


@dataclass()
class Tahn(ActivationLayer):
    def activate(self, val: float) -> float:
        return (np.exp(val) - np.exp(-val)) / (np.exp(val) + np.exp(-val))


@dataclass()
class ReLU(ActivationLayer):
    def activate(self, val: float) -> float:
        return max(0, val)


@dataclass()
class ParametricReLU(ActivationLayer):
    slope: float = 0.1

    def activate(self, val: float) -> float:
        return max(self.slope * val, val)


@dataclass()
class ELU(ActivationLayer):
    slope: float = 1

    def activate(self, val: float) -> float:
        return val if val >= 0 else self.slope * (np.exp(val) - 1)


@dataclass()
class Swish(ActivationLayer):
    def activate(self, val: float) -> float:
        return val / (1 + np.exp(-val))


@dataclass()
class SELU(ActivationLayer):
    slope: float = 1.0

    def activate(self, val: float) -> float:
        return val if val >= 0 else self.slope * (np.exp(val) - 1)
