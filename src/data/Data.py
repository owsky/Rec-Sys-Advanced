from typing import Literal
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
from scipy.sparse import coo_array, csr_array, csc_array


class Data:
    """
    Data class that wraps the ratings and item features and provides a single source of truth
    alongside useful methods for transforming the data
    """

    def __init__(
        self,
        movies_path: str,
        ratings_path: str,
        ratings_test_size: float | None = None,
    ):
        loaded = self._load_ratings(ratings_path, ratings_test_size)
        if isinstance(loaded, tuple):
            self.ratings = loaded[0]
            self.test_ratings = loaded[1]
        else:
            self.ratings = loaded

        self.items = self._load_movies(movies_path)

    def _read_ratings_file(self, ratings_path: str) -> NDArray[np.float64]:
        """
        Read file into memory converting strings to integers and return a numpy array
        """
        with open(ratings_path, "r") as f:
            ratings = []
            for line in f:
                user_id, item_id, rating, timestamp = line.rstrip().split("\t")
                ratings.append(
                    [int(user_id), int(item_id), int(rating), int(timestamp)]
                )
            f.close()
        ratings = np.array(ratings)
        return ratings

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

    def _create_ratings_matrix(
        self, ratings: NDArray[np.float64], shape: tuple[int, int]
    ):
        """
        Given a ratings array and a shape, create a sparse array in COO format
        """
        data = []
        row_indices = []
        col_indices = []

        for user, item, rating, _ in ratings:
            user_index = self.id_to_index(user, "user")
            item_index = self.id_to_index(item, "movie")
            data.append(rating)
            row_indices.append(user_index)
            col_indices.append(item_index)
        sorted_indices = np.lexsort((col_indices, row_indices))
        data = np.array(data)[sorted_indices]
        row_indices = np.array(row_indices)[sorted_indices]
        col_indices = np.array(col_indices)[sorted_indices]
        ratings_matrix = coo_array((data, (row_indices, col_indices)), shape=shape)
        return ratings_matrix

    def _handle_id_ranges(self, ratings: NDArray[np.float64]) -> tuple[int, int]:
        """
        Compute the shape that the user-item matrices will need to assume.
        This is needed in case some item ids happen to be <= 0. In such case
        an offset is introduced to account for this.
        """
        min_rows = ratings[:, 0].min()
        min_cols = ratings[:, 1].min()
        max_rows = ratings[:, 0].max()
        max_cols = ratings[:, 1].max()

        self.offset_rows = abs(min_rows) + 1 if min_rows <= 0 else 0
        self.offset_cols = abs(min_cols) + 1 if min_cols <= 0 else 0

        return max_rows + self.offset_rows, max_cols + self.offset_cols

    def _compute_average_rating(
        self, ratings: csr_array | csc_array, axis: Literal[0, 1]
    ):
        n_rows, n_cols = ratings.shape
        dim = n_rows if axis == 0 else n_cols
        means = []
        for index in range(dim):
            user_ratings = ratings[[index], :] if axis == 0 else ratings[:, [index]]
            nz = user_ratings.count_nonzero()
            if nz == 0:
                nz = 1
            user_mean = user_ratings.sum() / nz
            means.append(user_mean)
        return np.array(means)

    def _load_ratings(
        self, ratings_path: str, test_size: float | None = None
    ) -> tuple[coo_array, coo_array] | coo_array:
        """
        Load the ratings from file, optionally partition into train and test datasets by timestamp,
        and then create the user-item ratings matrices with sparse representations.
        """
        ratings = self._read_ratings_file(ratings_path)

        ratings_matrix_shape = self._handle_id_ranges(ratings)

        if test_size is not None:
            train, test = self._train_test_split(ratings, test_size)
            train_matrix = self._create_ratings_matrix(train, ratings_matrix_shape)
            test_matrix = self._create_ratings_matrix(test, ratings_matrix_shape)
            self.average_user_rating = self._compute_average_rating(
                train_matrix.tocsr(), 0
            )
            self.average_item_rating = self._compute_average_rating(
                train_matrix.tocsc(), 1
            )

            return train_matrix, test_matrix
        else:
            matrix = self._create_ratings_matrix(ratings, ratings_matrix_shape)
            self.average_user_rating = self._compute_average_rating(matrix.tocsr(), 0)
            self.average_item_rating = self._compute_average_rating(matrix.tocsc(), 1)
            return matrix

    def id_to_index(self, id: int, kind: Literal["user", "movie"]) -> int:
        """
        Convert an ID to an index of the user-item matrix
        """
        offset = self.offset_rows if kind == "user" else self.offset_cols
        return id - offset - 1

    def index_to_id(self, index: int, kind: Literal["user", "movie"]) -> int:
        """
        Convert an index of the user-item matrix to an ID
        """
        offset = self.offset_rows if kind == "user" else self.offset_cols
        return index + offset + 1

    def _load_movies(self, movies_path: str) -> DataFrame:
        """
        Load the movies information as DataFrame
        """
        col_names = [
            "movie_id",
            "movie_title",
            "release_date",
            "video_release_date",
            "IMDb_URL",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]
        self.genres_labels = col_names[5:]
        data = pd.read_csv(
            movies_path,
            sep="|",
            header=None,
            names=col_names,
            encoding="iso-8859-1",
            on_bad_lines="warn",
        )
        data = data.drop(["video_release_date", "IMDb_URL"], axis=1)
        return data

    def get_movies_from_ids(self, movie_ids: int | NDArray[np.int64]) -> DataFrame:
        """
        Given a movie id or an array of ids, retrieve the full movies information from
        the dataframe, sorted by the order of the provided ids
        """
        items_df = self.items
        indices = [movie_ids] if isinstance(movie_ids, int) else movie_ids.flatten()
        result_df = items_df[items_df["movie_id"].isin(indices)]
        result_df = result_df.set_index("movie_id")
        result_df = result_df.reindex(indices)
        result_df = result_df.reset_index()
        return result_df

    def get_movie_from_index(self, movie_indices: int | list[int]) -> DataFrame:
        """
        Given a movie index, return the movie information
        """
        l = []
        if isinstance(movie_indices, list):
            for index in movie_indices:
                l.append(self.index_to_id(index, "movie"))
        return self.get_movies_from_ids(np.array(l))

    def pretty_print_movies_df(self, movies_df: DataFrame) -> None:
        """
        Given a movies dataframe pretty print to standard output the movies information
        """
        df_genres = pd.DataFrame(
            {
                "ID": movies_df["movie_id"],
                "Title": movies_df["movie_title"],
                "Release Date": movies_df["release_date"],
                "Genres": movies_df[self.genres_labels].apply(
                    lambda row: ", ".join(
                        [
                            genre
                            for genre, value in zip(self.genres_labels, row)
                            if value == 1
                        ]
                    ),
                    axis=1,
                ),
            }
        )
        pd.set_option("colheader_justify", "center")
        print(df_genres)
