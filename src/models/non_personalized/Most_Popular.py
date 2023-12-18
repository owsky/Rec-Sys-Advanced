import numpy as np
from data import Data
from ..Recommender_System import Recommender_System
from scipy.sparse import csr_array


class Most_Popular(Recommender_System):
    def __init__(self, data: Data):
        super().__init__(data, "Most Popular")

    def fit(self, silent=False):
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
        user_id = self.data.user_index_to_id[user_index]
        ratings = self.data.get_user_ratings(user_id, "train")
        unrated_movies = np.flatnonzero(ratings == 0)

        z = zip(unrated_movies, self.popularity[unrated_movies])
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
