from typing import Literal
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
from scipy.sparse import coo_array


class Data:
    """
    Data class that wraps the ratings and item features and provides a single source of truth
    alongside useful methods for transforming the data
    """

    def __init__(
        self,
        movies_path: str,
        ratings_path: str,
        tags_path: str,
        ratings_test_size: float = 0.2,
    ):
        self._load_ratings(ratings_path, ratings_test_size)
        self._load_movies(movies_path)
        self._load_tags(tags_path)

    def _train_test_split(
        self, ratings: NDArray[np.float64], test_size: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Split ratings dataset into train and test according to test size and timestamps
        """
        ratings[ratings[:, 3].argsort()]  # Sort by timestamp

        n_rows = ratings.shape[0]
        split_index = int(n_rows * (1 - test_size))

        train = ratings[:split_index, :]
        test = ratings[split_index:, :]

        return train, test

    def _create_ratings_matrix(self, ratings: DataFrame, shape: tuple[int, int]):
        """
        Given a ratings array and a shape, create a sparse array in COO format
        """
        data = []
        indices = []

        for _, row in ratings.iterrows():
            user_id, item_id, rating = row["userId"], row["movieId"], row["rating"]

            if user_id in self.user_id_to_index:
                user_index = self.user_id_to_index[user_id]
            else:
                user_index = self.new_user_index
                self.user_id_to_index[user_id] = user_index
                self.user_index_to_id[user_index] = user_id
                self.new_user_index += 1

            if item_id in self.item_id_to_index:
                item_index = self.item_id_to_index[item_id]
            else:
                item_index = self.new_item_index
                self.item_id_to_index[item_id] = item_index
                self.item_index_to_id[item_index] = item_id
                self.new_item_index += 1

            data.append(rating)
            indices.append((user_index, item_index))

        sorted_indices = np.lexsort((np.array(indices)[:, 1], np.array(indices)[:, 0]))
        data = np.array(data)[sorted_indices]
        indices = np.array(indices)[sorted_indices]
        ratings_matrix = coo_array((data, indices.T), shape=shape)

        return ratings_matrix

    def _compute_average_ratings(self, coo_array: coo_array):
        num_rows = coo_array.shape[0]
        num_cols = coo_array.shape[1]

        row_sum_array = np.zeros(num_rows)
        row_count_array = np.zeros(num_rows)

        col_sum_array = np.zeros(num_cols)
        col_count_array = np.zeros(num_cols)

        for row, col, value in zip(coo_array.row, coo_array.col, coo_array.data):
            row_sum_array[row] += value
            if value != 0:
                row_count_array[row] += 1

            col_sum_array[col] += value
            if value != 0:
                col_count_array[col] += 1

        # Avoid division by zero
        row_count_array[row_count_array == 0] = 1
        col_count_array[col_count_array == 0] = 1

        row_averages = row_sum_array / row_count_array
        col_averages = col_sum_array / col_count_array

        return row_averages, col_averages

    def _load_ratings(
        self, ratings_path: str, test_size: float
    ) -> tuple[coo_array, coo_array]:
        """
        Load the ratings from file, optionally partition into train and test datasets by timestamp,
        and then create the user-item ratings matrices with sparse representations.
        """
        df = pd.read_csv(ratings_path)

        shape = (df["userId"].nunique(), df["movieId"].nunique())

        df = df.sort_values(by="timestamp")
        df = df.drop(columns=["timestamp"])
        split_index = int(test_size * len(df))
        train = df.iloc[:split_index]
        test = df.iloc[split_index:]

        self.user_id_to_index = {}
        self.user_index_to_id = {}
        self.new_user_index = 0
        self.item_id_to_index = {}
        self.item_index_to_id = {}
        self.new_item_index = 0

        self.train = self._create_ratings_matrix(train, shape)
        self.test = self._create_ratings_matrix(test, shape)

        # self.train = train.pivot(index="userId", columns="movieId", values="rating")
        (
            self.average_user_rating,
            self.average_item_rating,
        ) = self._compute_average_ratings(self.train)

        return self.train, self.test

    def id_to_index(self, id: int, kind: Literal["user", "item"]) -> int:
        """
        Convert an ID to an index of the user-item matrix
        """
        if kind == "user":
            return self.user_id_to_index[id]
        else:
            return self.item_id_to_index[id]

    def index_to_id(self, index: int, kind: Literal["user", "item"]) -> int:
        """
        Convert an index of the user-item matrix to an ID
        """
        if kind == "user":
            return self.user_index_to_id[index]
        else:
            return self.item_index_to_id[index]

    def _load_movies(self, movies_path: str) -> None:
        """
        Load the movies information as DataFrame
        """
        self.movies = pd.read_csv(movies_path)

        genres_df = self.movies["genres"].str.split("|", expand=True).stack()

        if isinstance(genres_df, pd.DataFrame):
            raise RuntimeError(
                "Something wrong happened while loading the movies dataset"
            )

        genres_df = genres_df.to_frame("unique_genres")
        self.genre_labels = genres_df["unique_genres"].unique().tolist()

    def _load_tags(self, tags_path: str) -> None:
        tags_df = pd.read_csv(tags_path)
        self.tags = tags_df.drop(columns="timestamp")

    def get_movies_from_ids(self, movie_ids: list[int]) -> DataFrame:
        """
        Given a movie id or an array of ids, retrieve the full movies information from
        the dataframe, sorted by the order of the provided ids
        """
        result_df = self.movies[self.movies["movieId"].isin(movie_ids)]
        result_df = result_df.set_index("movieId")
        result_df = result_df.reindex(movie_ids)
        result_df = result_df.reset_index()
        return result_df

    def get_movie_from_indices(self, movie_indices: list[int]) -> DataFrame:
        """
        Given a movie index, return the movie information
        """
        return self.get_movies_from_ids(
            [self.index_to_id(i, "item") for i in movie_indices]
        )
