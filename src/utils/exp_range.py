from typing import Literal
import numpy as np


def exp_range(
    start: int | float,
    stop: int | float,
    n_points: int,
    dtype: type[np.int64] | type[np.float64],
):
    return np.logspace(
        np.log10(start),
        np.log10(stop),
        num=n_points,
        endpoint=True,
        base=10.0,
        dtype=dtype,
    )
