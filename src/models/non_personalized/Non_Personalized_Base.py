from abc import ABC
from scipy.sparse import csr_array, csc_array
import numpy as np
from ..Recommender_System import Recommender_System


class Non_Personalized_Base(Recommender_System, ABC):
    """
    Non personalized recommender system
    """

    ratings_train: csc_array | None

    def _get_unrated_movies(self, user_index: int):
        if not self.is_fit:
            raise RuntimeError("Model untrained, invoke fit before predicting")

        user_id = self.data.user_index_to_id[user_index]
        user_row = self.data.get_user_ratings(user_id, "train")
        unrated_movies = np.flatnonzero(user_row == 0)

        if len(unrated_movies) == 0:
            unrated_movies = np.array(range(user_row.shape[1]))
        return unrated_movies

    def predict(self):
        raise RuntimeError(f"Model {self.__class__.__name__} cannot predict ratings")

    def _predict_all(self):
        raise RuntimeError(f"Model {self.__class__.__name__} cannot predict ratings")

    def crossvalidation_hyperparameters(self):
        raise RuntimeError(
            f"Model {self.__class__.__name__} has no hyperparameters to crossvalidate"
        )
