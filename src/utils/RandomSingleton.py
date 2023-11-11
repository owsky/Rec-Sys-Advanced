import random
from typing import AbstractSet, Sequence

import numpy as np


class RandomSingleton:
    _rng = None

    @staticmethod
    def initialize(seed: int | None = None):
        if RandomSingleton._rng is None:
            random.seed(seed)
            RandomSingleton._rng = np.random.default_rng(seed)

    @staticmethod
    def get_random_sample(population: Sequence | AbstractSet, k: int):
        RandomSingleton.initialize()
        return random.sample(population, k)

    @staticmethod
    def get_random_normal(loc: int, scale: float, size: tuple[int, int]):
        RandomSingleton.initialize()
        if RandomSingleton._rng is not None:
            return RandomSingleton._rng.normal(loc, scale, size)
        else:
            raise RuntimeError("RNG not initialized")

    @staticmethod
    def shuffle(data):
        RandomSingleton.initialize()
        if RandomSingleton._rng is not None:
            RandomSingleton._rng.shuffle(data)
        else:
            raise RuntimeError("RNG not initialized")
