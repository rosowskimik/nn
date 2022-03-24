from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from typing import Optional
from pathlib import Path
import jsonpickle

import numpy as np
from nn.genetic.pickers import Picker
from nn.layers.base import BaseLayer

from nn.network import NeuralNetwork


@dataclass()
class NetworkPool:
    networks: list[NeuralNetwork]

    picker: Picker

    generation: int = 1

    def __init__(
        self,
        pool_size: int,
        picker: Picker,
        structure: list[BaseLayer],
        generation: int = 1,
    ):
        self.networks = list()
        self.picker = picker
        self.generation = generation

        for _ in range(pool_size):
            cp = deepcopy(structure)

            for layer in cp:
                layer.randomize()

            self.networks.append(NeuralNetwork(cp))

    def forward(self, index: int, inputs: np.ndarray) -> np.ndarray:
        return self.networks[index].forward(inputs)

    def forward_all(self, inputs: np.ndarray) -> np.ndarray:
        return np.array(
            [network.forward(inputs[i]) for i, network in enumerate(self.networks)]
        )

    def next_generation(
        self, muation_rate: float, fitness: list[float], keep_best: int = 1
    ):
        assert keep_best <= len(self.networks)
        new_networks = list()

        self.picker.set_pool(fitness, self.networks)

        new_networks.extend(self.picker.best_n(keep_best))

        while len(new_networks) < len(self.networks):
            p1, p2 = self.picker.pick()

            crossed = p1.cross_with(p2)
            crossed.mutate_network(muation_rate)

            new_networks.append(crossed)

        self.generation += 1

    @staticmethod
    def from_json(json: str) -> "NetworkPool":
        return jsonpickle.decode(json)

    def to_json(self) -> str:
        return jsonpickle.encode(self)

    @staticmethod
    def from_file(file: PathLike) -> "NetworkPool":
        with open(file) as f:
            return jsonpickle.decode(f.read())

    def to_file(
        self,
        filename: str,
        directory: Optional[PathLike] = None,
        create_parents: bool = False,
        overwrite: bool = True,
    ):
        directory = Path.cwd() if directory is None else Path(directory)

        if not directory.exists():
            directory.mkdir(parents=create_parents)

        elif not directory.is_dir():
            raise NotADirectoryError

        target = directory.joinpath(filename + ".ntp")
        mode = "w" if overwrite else "x"

        with open(target, mode) as f:
            f.write(self.to_json())
