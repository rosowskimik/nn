from dataclasses import dataclass

import numpy as np
from gnn.layers.activators import ActivationLayer
from gnn.layers.base import BaseLayer
from gnn.layers.dense import DenseLayer
from gnn.layers.softmax import SoftmaxLayer


@dataclass()
class CombinedLayer(BaseLayer):
    """Layer, which combines dense, activation and optionally softmax layers."""

    dense: DenseLayer
    activation: ActivationLayer

    use_softmax: bool = False

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        output = self.dense.forward(inputs)
        output = self.activation.forward(inputs)

        if self.use_softmax:
            output = SoftmaxLayer().forward(output)

        return output
