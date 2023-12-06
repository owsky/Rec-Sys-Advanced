from typing_extensions import Self
import numpy as np
from data import Data
from .Non_Personalized_Base import Non_Personalized_Base


class Highest_Rated(Non_Personalized_Base):
    def __init__(self):
        super().__init__("Highest Rated")

    def fit(self, data: Data) -> Self:
        self.data = data
        self.ratings_train = data.train.tocsc()
        self.average_ratings = self.ratings_train.mean(axis=0)
        return self

    def top_n(self, user_index: int, n: int):
        unrated_movies = self._get_unrated_movies(user_index)
        # Find the indices of the top n highest average values for unrated items
        top_n_indices = np.argpartition(self.average_ratings[unrated_movies], -n)[-n:]
        top_n_indices = unrated_movies[top_n_indices]

        # Return the top n highest rated unrated items
        movie_ids = np.array(
            [self.data.index_to_id(idx, "item") for idx in top_n_indices]
        )
        return movie_ids
