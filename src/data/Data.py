from typing import Dict, Literal
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
from scipy.sparse import coo_array, csr_array
from utils import exponential_decay


class Data:
    """
    Data class that wraps the ratings and item features and provides a single source of truth
    alongside useful methods
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

    def _compute_average_ratings(
        self, ratings: coo_array
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute for each user and item the average non-zero rating
        """
        n_rows, n_cols = ratings.shape

        row_sum_array = np.zeros(n_rows)
        row_count_array = np.zeros(n_rows)

        col_sum_array = np.zeros(n_cols)
        col_count_array = np.zeros(n_cols)

        for user_index, item_index, rating in zip(
            ratings.row, ratings.col, ratings.data
        ):
            row_sum_array[user_index] += rating
            if rating != 0:
                row_count_array[user_index] += 1

            col_sum_array[item_index] += rating
            if rating != 0:
                col_count_array[item_index] += 1

        # Avoid division by zero
        row_count_array[row_count_array == 0] = 1
        col_count_array[col_count_array == 0] = 1

        row_averages = row_sum_array / row_count_array
        col_averages = col_sum_array / col_count_array

        return row_averages, col_averages

    def _create_interactions(
        self, ratings: DataFrame, shape: tuple[int, int], test_size: float
    ):
        """
        Given the ratings dataframe, the correct shape and the test size, compute
        the interactions matrices for both train and test.
        Also create dataframes for the split data and dictionaries which convert
        from ids to indices and vice versa.
        """
        data_train = []
        row_indices_train = []
        col_indices_train = []
        data_test = []
        row_indices_test = []
        col_indices_test = []
        new_user_index = 0

        self.user_id_to_index: Dict[int, int] = {}
        self.user_index_to_id: Dict[int, int] = {}

        self.ratings_train_df = pd.DataFrame()
        self.ratings_test_df = pd.DataFrame()

        def process_user(df: DataFrame, kind: Literal["train", "test"]):
            nonlocal new_user_index
            for _, row in df.iterrows():
                movie_id, r = row["movieId"], row["rating"]

                if user_id in self.user_id_to_index:
                    user_index = self.user_id_to_index[user_id]
                else:
                    user_index = new_user_index
                    self.user_id_to_index[user_id] = user_index
                    self.user_index_to_id[user_index] = user_id
                    new_user_index += 1

                item_index = self.item_id_to_index[movie_id]
                if kind == "train":
                    data_train.append(r)
                    row_indices_train.append(user_index)
                    col_indices_train.append(item_index)
                else:
                    data_test.append(r)
                    row_indices_test.append(user_index)
                    col_indices_test.append(item_index)

        for user_id in ratings["userId"].unique():
            user_df = ratings[ratings["userId"] == user_id]

            split_index = int((1 - test_size) * len(user_df))

            user_train: DataFrame = user_df.iloc[:split_index]
            self.ratings_train_df = pd.concat(
                [self.ratings_train_df, user_train], ignore_index=True
            )
            process_user(user_train, "train")

            user_test: DataFrame = user_df.iloc[split_index:]
            self.ratings_test_df = pd.concat(
                [self.ratings_test_df, user_test], ignore_index=True
            )
            process_user(user_test, "test")

        return coo_array(
            (data_train, (row_indices_train, col_indices_train)), shape=shape
        ), coo_array((data_test, (row_indices_test, col_indices_test)), shape=shape)

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
                "timestamp": "Int64",
            },
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.sort_values(by="timestamp", inplace=True)
        shape = (df["userId"].nunique(), self.how_many_unique_movie_ids)

        self.interactions_train, self.interactions_test = self._create_interactions(
            df, shape, test_size
        )
        self.interactions_train_numpy = self.interactions_train.toarray()

        (
            self.average_user_rating,
            self.average_item_rating,
        ) = self._compute_average_ratings(self.interactions_train)

        return self.interactions_train, self.interactions_test

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
    ) -> NDArray:
        user_index = self.user_id_to_index[user_id]
        arr = self.interactions_train if dataset == "train" else self.interactions_test
        return csr_array(arr.getrow(user_index)).toarray()[0]

    def get_weighed_user_ratings(self, user_id: int):
        user_df = self.ratings_train_df[self.ratings_train_df["userId"] == user_id]
        timestamps = user_df["timestamp"]
        try:
            base_time = timestamps.tail(1).item()
        except ValueError:
            return []
        weights = [exponential_decay(timestamp, base_time) for timestamp in timestamps]
        ratings: list[tuple[int, float]] = list(
            user_df[["movieId", "rating"]].itertuples(index=False, name=None)
        )
        return [(tup[0], tup[1], weight) for tup, weight in zip(ratings, weights)]

    def get_liked_movies_indices(
        self, user_id: int, biased: bool, dataset: Literal["train", "test"]
    ) -> list[int]:
        user_ratings = self.get_user_ratings(user_id, dataset)
        nz = user_ratings.nonzero()
        if len(user_ratings[nz]) == 0:
            return []
        user_index = self.user_id_to_index[user_id]
        if biased:
            user_bias = (
                0
                if np.std(user_ratings[nz]) == 0
                else self.average_user_rating[user_index]
            )
            mask = user_ratings - user_bias >= 0
        else:
            mask = user_ratings >= 3
        liked_indices = np.flatnonzero(mask)
        return sorted(liked_indices, key=lambda x: user_ratings[x], reverse=True)

    def get_user_bias(self, user_id: int):
        user_index = self.user_id_to_index[user_id]
        user_ratings = self.get_user_ratings(user_id, "train")
        if np.count_nonzero(user_ratings) == 0:
            return 0
        nz = user_ratings.nonzero()
        user_bias = (
            0 if np.std(user_ratings[nz]) == 0 else self.average_user_rating[user_index]
        )
        return user_bias

    def get_weighed_liked_movie_indices(self, user_id: int, biased: bool):
        user_ratings = self.get_weighed_user_ratings(user_id)
        if len(user_ratings) == 0:
            return []
        user_bias = self.get_user_bias(user_id)

        condition = lambda r: r - user_bias >= 0.0 if biased else lambda r: r >= 3.0

        user_likes = sorted(
            [
                (self.item_id_to_index[movie_id], rating, weight)
                for (movie_id, rating, weight) in user_ratings
                if condition(rating)
            ],
            key=lambda x: x[2],
            reverse=True,
        )
        return user_likes

    def get_ratings_count(self, user_id: int):
        ratings = self.get_user_ratings(user_id, "train")
        return np.count_nonzero(ratings)
