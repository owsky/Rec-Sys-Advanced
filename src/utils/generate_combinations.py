# from itertools import product
from tqdm.contrib.itertools import product
from typing import Dict
from numpy.typing import NDArray


def generate_combinations(data: Dict[str, list] | Dict[str, NDArray]):
    keys = list(data.keys())
    value_lists = [data[key] for key in keys]
    for combination in product(*value_lists):
        yield dict(zip(keys, combination))
