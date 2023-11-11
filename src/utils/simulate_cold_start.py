from typing import Literal
from scipy.sparse import coo_array, lil_array
from .RandomSingleton import RandomSingleton


def simulate_cold_start(R: coo_array, user_perc=0.1, item_perc=0) -> coo_array:
    R_lil: lil_array = R.tolil()
    users, items = R_lil.shape
    users_to_purge = int(users * user_perc)
    items_to_purge = int(items * item_perc)

    rng = RandomSingleton()

    def purge(how_many: int, axis: Literal[0, 1]):
        tp = rng.get_random_sample(range(how_many), how_many)
        if axis == 0:
            R_lil[tp, :] = 0
        else:
            R_lil[:, tp] = 0

    purge(users_to_purge, 0)
    purge(items_to_purge, 1)
    return R_lil.tocoo()
