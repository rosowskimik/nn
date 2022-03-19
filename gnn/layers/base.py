from dataclasses import dataclass

import jsonpickle
import numpy as np
from typing_extensions import Self


@dataclass()
class BaseLayer:
    """Base Layer class from which all layers inherit.
    Should not be instanciated on its own. All derived classes should implement `forward` method."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forwards input array through this layer."""
        raise NotImplementedError

    def to_json(self) -> str:
        """Converts layer to json."""
        return jsonpickle.encode(self)

    @staticmethod
    def from_json(json: str) -> Self:
        """Loads Layer from json."""
        return jsonpickle.decode(json)
