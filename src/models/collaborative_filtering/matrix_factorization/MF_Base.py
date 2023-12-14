from abc import ABC
import numpy as np
from numpy.typing import NDArray
from data import Data
from ..CF_Base import CF_Base


class MF_Base(CF_Base, ABC):
    """
    Base class for Matrix Factorization recommender systems
    """

    P: NDArray[np.float64]
    Q: NDArray[np.float64]

    def __init__(self, data: Data, model_name: str):
        super().__init__(data, model_name)

    def predict(self, u: int, i: int) -> float:
        if not self.is_fit:
            raise RuntimeError("Model untrained, invoke fit before predicting")
        prediction = np.dot(self.P[u, :], self.Q[i, :].T)
        return np.clip(prediction, 1, 5)

    def top_n(self, user_index: int, n=10):
        if not self.is_fit:
            raise RuntimeError("Model untrained, fit first")
        user_id = self.data.user_index_to_id[user_index]
        ratings = self.data.get_user_ratings(user_id, "train")
        unrated_indices = np.flatnonzero(ratings == 0)
        predictions = [
            (item_index, self.predict(user_index, item_index))
            for item_index in unrated_indices
        ]
        predictions = [
            self.data.item_index_to_id[x[0]]
            for x in sorted(predictions, key=lambda x: x[1], reverse=True)
        ]
        return predictions[:n]
