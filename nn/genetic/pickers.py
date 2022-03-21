from dataclasses import dataclass, field

from nn.network import NeuralNetwork


@dataclass()
class Picker:
    pool: list[tuple(float, NeuralNetwork)] = field(default_factory=list)

    def set_pool(self, fitness: list[float], networks: list[NeuralNetwork]):
        self.pool = sorted((zip(fitness, networks)), key=lambda x: x[0], reverse=True)

    def pick(self) -> tuple[NeuralNetwork, NeuralNetwork]:
        pass

    def best_n(self, n: int) -> list[NeuralNetwork]:
        return list(map(lambda x: x[1], self.pool[:n]))
