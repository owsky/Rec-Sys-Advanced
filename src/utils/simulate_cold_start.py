from typing import Literal
from scipy.sparse import coo_array, lil_array
from .RandomSingleton import RandomSingleton


def simulate_cold_start(train_set: coo_array, user_perc=0.1, item_perc=0) -> coo_array:
    """
    Given a train set and percentages of cold users and cold items to simulate,
    randomly selects users and items and sets all theirs ratings to zero
    """
    # Convert train set to LIL format for easier slicing
    train_set_lil: lil_array = train_set.tolil()
    users, items = train_set_lil.shape

    # Compute how many users and items to purge from train set
    users_to_purge = int(users * user_perc)
    items_to_purge = int(items * item_perc)

    rng = RandomSingleton()

    def purge(how_many: int, axis: Literal[0, 1]):
        tp = rng.get_random_sample(range(how_many), how_many)
        if axis == 0:
            train_set_lil[tp, :] = 0
        else:
            train_set_lil[:, tp] = 0

    # Purge ratings from both rows and columns
    purge(users_to_purge, 0)
    purge(items_to_purge, 1)
    return train_set_lil.tocoo()
