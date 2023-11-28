import numpy as np


def lists_str_join(l1: list[str], l2: list[str]):
    comb = list(set(np.array(l1).flatten().tolist() + np.array(l2).flatten().tolist()))
    merged = " ".join(comb)
    return merged
