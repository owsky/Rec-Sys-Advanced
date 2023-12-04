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
        if self.ratings_train is None:
            raise RuntimeError("Model untrained, invoke fit before predicting")
        user_row = csr_array(self.ratings_train.getrow(user_index)).toarray()
        unrated_movies = np.where(user_row == 0)[1]
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
