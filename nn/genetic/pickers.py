from dataclasses import dataclass, field

import numpy as np

from nn.network import NeuralNetwork


@dataclass()
class Picker:
    fitnesses: list[float] = field(default_factory=list)
    pool: list[NeuralNetwork] = field(default_factory=list)

    def set_pool(self, fitness: list[float], networks: list[NeuralNetwork]):
        for fit, network in sorted(
            (zip(fitness, networks)), key=lambda x: x[0], reverse=True
        ):
            self.fitnesses.append(fit)
            self.pool.append(network)

    def pick(self) -> tuple[NeuralNetwork, NeuralNetwork]:
        pass

    def best_n(self, n: int) -> list[NeuralNetwork]:
        return self.pool[:n]


@dataclass()
class RandomPicker(Picker):
    def pick(self) -> tuple[NeuralNetwork, NeuralNetwork]:
        n1, n2 = np.random.choice(self.pool, 2, replace=False)
        return tuple(n1, n2)


@dataclass()
class WeightedPicker(Picker):
    def pick(self) -> tuple[NeuralNetwork, NeuralNetwork]:
        n1, n2 = np.random.choice(self.pool, 2, replace=False, p=self.fitnesses)
        return tuple(n1, n2)


@dataclass()
class BestPicker(Picker):
    def pick(self) -> tuple[NeuralNetwork, NeuralNetwork]:
        n1, n2 = self.best_n(2)
        return tuple(n1, n2)
