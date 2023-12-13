import numpy as np
from data import Data
from typing_extensions import Self
from .Non_Personalized_Base import Non_Personalized_Base
from scipy.sparse import csr_array


class Most_Popular(Non_Personalized_Base):
    def __init__(self, data: Data):
        super().__init__(data, "Most Popular")

    def fit(self, silent=False) -> Self:
        if not silent:
            print("Fitting Most Popular model")
        self.is_fit = True
        popularity = []
        n_users, n_items = self.data.interactions_train.shape
        for item_index in range(n_items):
            item_ratings = csr_array(self.data.interactions_train.getcol(item_index))
            nz = item_ratings.count_nonzero()
            popularity.append(nz / n_users)
        self.popularity = np.array(popularity)
        return self

    def top_n(self, user_index: int, n: int):
        unrated_movies = self._get_unrated_movies(user_index)

        z = zip(unrated_movies, self.popularity[unrated_movies])
        z = sorted(z, key=lambda x: x[1], reverse=True)
        top_n_indices = [x[0] for x in z][:n]

        movie_ids = np.array(
            [self.data.index_to_id(idx, "item") for idx in top_n_indices]
        )
        return movie_ids
