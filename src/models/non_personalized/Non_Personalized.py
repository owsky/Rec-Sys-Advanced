from scipy.sparse import csr_array
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from data import Data


class Non_Personalized:
    """
    Non personalized recommendations. Provides top N most popular and top N highest rated
    """

    def fit(self, data: Data):
        self.data = data
        self.ratings = self.data.train.tocsc()
        self.average_ratings = self.ratings.mean(axis=0)

        pop = []
        n_rows, n_cols = self.ratings.shape
        for col_idx in range(n_cols):
            col = self.ratings[:, [col_idx]]
            nz = col.count_nonzero()
            pop.append(nz / n_rows)
        self.popularity = np.array(pop)

    def _get_unrated_movies(self, user_id: int):
        user_index = self.data.id_to_index(user_id, "user")
        user_row = csr_array(self.ratings.getrow(user_index)).toarray()
        unrated_movies = np.where(user_row == 0)[1]
        if len(unrated_movies) == 0:
            unrated_movies = np.array(range(user_row.shape[1]))
        return unrated_movies

    def _get_top_n(self, user_id: int, n: int, precomputed_metric: NDArray):
        unrated_movies = self._get_unrated_movies(user_id)
        arr = precomputed_metric[unrated_movies]
        # Find the indices of the top n highest average values for unrated items
        top_n_indices = np.argpartition(arr, -n)[-n:]
        top_n_indices = unrated_movies[top_n_indices]

        # Return the top n highest rated unrated items
        movie_ids = np.array(
            [self.data.index_to_id(idx, "movie") for idx in top_n_indices]
        )
        movies = self.data.get_movies_from_ids(movie_ids)
        return movies

    def get_n_highest_rated(self, user_id: int, n=10) -> DataFrame:
        return self._get_top_n(user_id, n, self.average_ratings)

    def get_n_most_popular(self, user_id: int, n=10) -> DataFrame:
        return self._get_top_n(user_id, n, self.popularity)
