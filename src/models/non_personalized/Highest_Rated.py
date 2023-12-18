import numpy as np
from data import Data
from ..Recommender_System import Recommender_System


class Highest_Rated(Recommender_System):
    def __init__(self, data: Data):
        super().__init__(data, "Highest Rated")

    def fit(self):
        self.is_fit = True
        return self

    def top_n(self, user_index: int, n: int):
        user_id = self.data.user_index_to_id[user_index]
        ratings = self.data.get_user_ratings(user_id, "train")
        unrated_movies = np.flatnonzero(ratings == 0)

        z = zip(unrated_movies, self.data.average_item_rating[unrated_movies])
        z = sorted(z, key=lambda x: x[1], reverse=True)
        top_n_indices = [x[0] for x in z][:n]

        movie_ids = np.array([self.data.item_index_to_id[idx] for idx in top_n_indices])
        return movie_ids

    def predict(self):
        raise RuntimeError(f"Model {self.__class__.__name__} cannot predict ratings")

    def _predict_all(self):
        raise RuntimeError(f"Model {self.__class__.__name__} cannot predict ratings")

    def gridsearch_cv(self):
        raise RuntimeError(
            f"Model {self.__class__.__name__} has no hyperparameters to crossvalidate"
        )
