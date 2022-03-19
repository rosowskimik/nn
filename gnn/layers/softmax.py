import numpy as np
from gnn.layers.base import BaseLayer


class SoftmaxLayer(BaseLayer):
    """A layer, which normalized all of its inputs, so that they sum up to 1.0."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        exponentiated = np.exp(inputs)
        return exponentiated / exponentiated.sum()
