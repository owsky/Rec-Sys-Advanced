from typing import Dict, Literal
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
from scipy.sparse import coo_array, csr_array

import data


class Data:
    """
    Data class that wraps the ratings and item features and provides a single source of truth
    alongside useful methods for transforming the data
    """

    def __init__(
        self,
        data_path: str,
        ratings_test_size: float = 0.2,
    ):
        self.data_path = data_path
        print("Loading the datasets...")
        self._load_movies()
        self._load_ratings(ratings_test_size)
        print("Datasets loaded correctly")

    def _create_ratings_matrix(self, ratings: DataFrame, shape: tuple[int, int]):
        """
        Given a ratings array and a shape, create a sparse array in COO format
        """
        data = []
        row_indices = []
        col_indices = []

        for _, row in ratings.iterrows():
            user_id, item_id, rating = row["userId"], row["movieId"], row["rating"]

            if user_id in self.user_id_to_index:
                user_index = self.user_id_to_index[user_id]
            else:
                user_index = self.new_user_index
                self.user_id_to_index[user_id] = user_index
                self.user_index_to_id[user_index] = user_id
                self.new_user_index += 1

            item_index = self.item_id_to_index[item_id]

            data.append(rating)
            row_indices.append(user_index)
            col_indices.append(item_index)

        ratings_matrix = coo_array((data, (row_indices, col_indices)), shape=shape)

        return ratings_matrix

    def _compute_average_ratings(
        self, coo_array: coo_array
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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

    def _load_ratings(self, test_size: float) -> tuple[coo_array, coo_array]:
        """
        Load the ratings from file, optionally partition into train and test datasets by timestamp,
        and then create the user-item ratings matrices with sparse representations.
        """
        df = pd.read_csv(
            self.data_path + "ratings.csv",
            dtype={
                "movie_id": "Int64",
                "userId": "Int64",
                "rating": float,
                "timestamp": int,
            },
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        movie_ids_with_tags = self.movies["movieId"].values
        df = df[df["movieId"].isin(movie_ids_with_tags)]

        shape = (df["userId"].nunique(), self.how_many_unique_movie_ids)

        df = df.sort_values(by="timestamp")
        split_index = int(test_size * (1 - len(df)))
        train = df.iloc[:split_index]
        self.train_ratings_df = train
        test = df.iloc[split_index:]

        self.user_id_to_index: Dict[int, int] = {}
        self.user_index_to_id: Dict[int, int] = {}
        self.new_user_index = 0

        self.train = self._create_ratings_matrix(train, shape)
        self.test = self._create_ratings_matrix(test, shape)

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

    def _load_movies(self) -> None:
        """
        Load the movies information as DataFrame
        """
        movies_df = pd.read_csv(
            self.data_path + "movies.csv",  # "movies_filtered.csv",
            dtype={"movieId": int, "title": str, "genres": str},
        )

        movies_df["genres"] = movies_df["genres"].str.lower()

        tags = pd.read_csv(
            self.data_path + "tags.csv",
            dtype={"tagId": int, "movieId": int, "tag": str},
        )
        tags["tag"] = tags["tag"].str.lower()

        merged = pd.merge(movies_df, tags, on="movieId", how="inner")
        self.movies = (
            merged.groupby(["movieId", "title", "genres"])["tag"]
            .agg(list)
            .reset_index()
            .rename(columns={"tag": "tags"})
        )

        all_movie_ids = self.movies["movieId"].unique()

        self.item_index_to_id = {
            index: movie_id for index, movie_id in enumerate(all_movie_ids)
        }
        self.item_id_to_index = {
            movie_id: index for index, movie_id in self.item_index_to_id.items()
        }

        self.how_many_unique_movie_ids = len(self.item_id_to_index.values())

        self.movies["genres"] = self.movies["genres"].str.split("|")
        self.genre_vocabulary = self.movies.explode("genres")["genres"].unique()
        self.tags_vocabulary = self.movies.explode("tags")["tags"].unique()
        self.max_num_genres = self.movies["genres"].apply(len).max()
        self.max_num_tags = self.movies["tags"].apply(len).max()

    def get_movies_from_ids(
        self, movie_ids: list[int] | NDArray[np.int64]
    ) -> DataFrame:
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

    def get_user_ratings(
        self, user_id: int, dataset: Literal["train", "test"]
    ) -> NDArray[np.int64]:
        user_index = self.user_id_to_index[user_id]
        arr = self.train if dataset == "train" else self.test
        return csr_array(arr.getrow(user_index)).toarray()[0]

    def get_liked_movies_indices(
        self, user_id: int, dataset: Literal["train", "test"]
    ) -> list[int]:
        user_ratings = self.get_user_ratings(user_id, dataset)
        nz = user_ratings.nonzero()
        if len(user_ratings[nz]) == 0:
            return []
        user_index = self.user_id_to_index[user_id]
        user_bias = (
            0 if np.std(user_ratings[nz]) == 0 else self.average_user_rating[user_index]
        )
        liked_indices = np.flatnonzero(user_ratings - user_bias > 0)
        return sorted(liked_indices, key=lambda x: user_ratings[x], reverse=True)
