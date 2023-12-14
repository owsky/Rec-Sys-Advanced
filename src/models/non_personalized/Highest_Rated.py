import numpy as np
from data import Data
from .Non_Personalized_Base import Non_Personalized_Base


class Highest_Rated(Non_Personalized_Base):
    def __init__(self, data: Data):
        super().__init__(data, "Highest Rated")

    def fit(self, silent=False):
        self.is_fit = True
        return self

    def top_n(self, user_index: int, n: int):
        unrated_movies = self._get_unrated_movies(user_index)

        z = zip(unrated_movies, self.data.average_item_rating[unrated_movies])
        z = sorted(z, key=lambda x: x[1], reverse=True)
        top_n_indices = [x[0] for x in z][:n]

        movie_ids = np.array(
            [self.data.index_to_id(idx, "item") for idx in top_n_indices]
        )
        return movie_ids
