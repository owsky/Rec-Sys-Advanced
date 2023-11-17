import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array


def _read_file(ratings_path: str) -> NDArray[np.float64]:
    with open(ratings_path, "r") as f:
        ratings = []
        for line in f:
            user_id, item_id, rating, timestamp = line.rstrip().split("\t")
            ratings.append([int(user_id), int(item_id), int(rating), int(timestamp)])
        f.close()
    ratings = np.array(ratings)
    return ratings


def _train_test_split(ratings: NDArray, test_size: float):
    ratings[ratings[:, 3].argsort()]  # Sort by timestamp

    rows = ratings.shape[0]
    indices = int(rows * test_size)

    train = ratings[:indices, :]
    test = ratings[indices:, :]
    return train, test


def _create_ratings_matrix(ratings: NDArray, shape: tuple[int, int]):
    ratings_matrix = np.zeros(shape)
    for user, item, rating, _ in ratings:
        ratings_matrix[user - 1][item - 1] = rating
    return ratings_matrix


def load_ratings(ratings_path: str, test_size=0.25) -> tuple[coo_array, coo_array]:
    """
    Load the ratings from file, partition into train and test datasets by timestamp
    and create the user-item ratings matrices with sparse representations.
    """
    ratings = _read_file(ratings_path)

    train, test = _train_test_split(ratings, test_size)

    ratings_matrix_shape = (ratings[:, 0].max(), ratings[:, 1].max())

    train_matrix = _create_ratings_matrix(train, ratings_matrix_shape)
    test_matrix = _create_ratings_matrix(test, ratings_matrix_shape)
    return coo_array(train_matrix), coo_array(test_matrix)
