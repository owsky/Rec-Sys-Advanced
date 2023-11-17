from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array


class RatingsReader:
    def _read_file(self, ratings_path: str) -> NDArray[np.float64]:
        with open(ratings_path, "r") as f:
            ratings = []
            ratings.append([-5, -3, 4, 881250949])
            for line in f:
                user_id, item_id, rating, timestamp = line.rstrip().split("\t")
                ratings.append(
                    [int(user_id), int(item_id), int(rating), int(timestamp)]
                )
            f.close()
        ratings = np.array(ratings)
        return ratings

    def _train_test_split(self, ratings: NDArray, test_size: float):
        ratings[ratings[:, 3].argsort()]  # Sort by timestamp

        rows = ratings.shape[0]
        indices = int(rows * test_size)

        train = ratings[:indices, :]
        test = ratings[indices:, :]
        return train, test

    def _create_ratings_matrix(self, ratings: NDArray, shape: tuple[int, int]):
        ratings_matrix = np.zeros(shape)
        for user, item, rating, _ in ratings:
            user_index = self.id_to_index(user, "user")
            item_index = self.id_to_index(item, "item")
            ratings_matrix[user_index][item_index] = rating
        return ratings_matrix

    def _handle_id_ranges(self, ratings: NDArray):
        min_rows = ratings[:, 0].min()
        min_cols = ratings[:, 1].min()
        max_rows = ratings[:, 0].max()
        max_cols = ratings[:, 1].max()

        self.offset_rows = abs(min_rows) + 1 if min_rows <= 0 else 0
        self.offset_cols = abs(min_cols) + 1 if min_cols <= 0 else 0

        return max_rows + self.offset_rows, max_cols + self.offset_cols

    def load_ratings(
        self, ratings_path: str, test_size: float | None = None
    ) -> tuple[coo_array, coo_array] | coo_array:
        """
        Load the ratings from file, optionally partition into train and test datasets by timestamp,
        and then create the user-item ratings matrices with sparse representations.
        """
        ratings = self._read_file(ratings_path)

        ratings_matrix_shape = self._handle_id_ranges(ratings)

        if test_size is not None:
            train, test = self._train_test_split(ratings, test_size)

            train_matrix = self._create_ratings_matrix(train, ratings_matrix_shape)
            test_matrix = self._create_ratings_matrix(test, ratings_matrix_shape)
            return coo_array(train_matrix), coo_array(test_matrix)
        else:
            matrix = self._create_ratings_matrix(ratings, ratings_matrix_shape)
            return coo_array(matrix)

    def id_to_index(self, id: int, kind: Literal["user", "item"]):
        offset = self.offset_rows if kind == "user" else self.offset_cols
        return id - offset - 1

    def index_to_id(self, index: int, kind: Literal["user", "item"]):
        offset = self.offset_rows if kind == "user" else self.offset_cols
        return index + offset + 1
