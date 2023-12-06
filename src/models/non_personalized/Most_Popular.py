import numpy as np
from data import Data
from typing_extensions import Self
from .Non_Personalized_Base import Non_Personalized_Base


class Most_Popular(Non_Personalized_Base):
    def __init__(self):
        super().__init__("Most Popular")

    def fit(self, data: Data) -> Self:
        self.data = data
        self.ratings_train = data.train.tocsc()
        pop = []
        n_rows, n_cols = self.ratings_train.shape
        for col_idx in range(n_cols):
            col = self.ratings_train[:, [col_idx]]
            nz = col.count_nonzero()
            pop.append(nz / n_rows)
        self.popularity = np.array(pop)
        return self

    def top_n(self, user_index: int, n: int):
        unrated_movies = self._get_unrated_movies(user_index)
        # Find the indices of the top n highest average values for unrated items
        top_n_indices = np.argpartition(self.popularity[unrated_movies], -n)[-n:]
        top_n_indices = unrated_movies[top_n_indices]

        # Return the top n highest rated unrated items
        movie_ids = np.array(
            [self.data.index_to_id(idx, "item") for idx in top_n_indices]
        )
        return movie_ids
