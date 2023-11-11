from scipy.sparse import coo_array


def check_cold_start(R: coo_array) -> tuple[int, int]:
    cold_users = sum(1 for _ in range(R.shape[0]) if R.getnnz(axis=1)[_] == 0)
    cold_items = sum(1 for _ in range(R.shape[1]) if R.getnnz(axis=0)[_] == 0)
    return cold_users, cold_items
