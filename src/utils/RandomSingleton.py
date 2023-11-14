import random
from typing import AbstractSet, Sequence
import numpy as np


class RandomSingleton:
    """
    Singleton class that wraps both Python and Numpy's random number generators, optionally with a seed
    """

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
        return RandomSingleton._rng.normal(loc, scale, size)  # type: ignore

    @staticmethod
    def shuffle(data):
        RandomSingleton.initialize()
        RandomSingleton._rng.shuffle(data)  # type: ignore
