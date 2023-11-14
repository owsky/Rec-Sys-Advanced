from scipy.sparse import coo_array


def check_cold_start(train_data: coo_array) -> tuple[int, int]:
    """
    Return how many cold users and cold items are present in the given data
    """
    cold_users = sum(
        1 for _ in range(train_data.shape[0]) if train_data.getnnz(axis=1)[_] == 0
    )
    cold_items = sum(
        1 for _ in range(train_data.shape[1]) if train_data.getnnz(axis=0)[_] == 0
    )
    return cold_users, cold_items
