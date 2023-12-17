from itertools import product
from typing import Dict
import numpy as np
from numpy.typing import NDArray


def generate_combinations(
    data: Dict[str, list[int | float] | NDArray[np.int64 | np.float64]]
):
    keys = list(data.keys())
    value_lists = [data[key] for key in keys]
    total = 1
    for l in value_lists:
        total *= len(l)

    def create_generator():
        for combination in product(*value_lists):
            yield dict(zip(keys, combination))

    return create_generator(), total
