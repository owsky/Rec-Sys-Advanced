from abc import ABC, abstractmethod
from itertools import product
from scipy.sparse import coo_array
import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate
from data import Data
from ..CF_Base import CF_Base
from typing_extensions import Self
from scipy.sparse import csr_array


class MF_Base(CF_Base, ABC):
    """
    Base class for Collaborative Filtering recommender systems
    """

    is_fit: bool

    def __init__(self, data: Data, model_name: str):
        super().__init__(data, model_name)
        self.P: NDArray[np.float64] = np.array([])
        self.Q: NDArray[np.float64] = np.array([])

    @abstractmethod
    def fit(self) -> Self:
        pass

    def predict(self, u: int, i: int) -> float:
        if self.P.size == 0 or self.Q.size == 0:
            raise RuntimeError("Model untrained, invoke fit before predicting")
        prediction = np.dot(self.P[u, :], self.Q[i, :].T)
        return np.clip(prediction, 1, 5)

    def top_n(self, user_index: int, n=10):
        if not self.is_fit:
            raise RuntimeError("Model untrained, fit first")
        user_id = self.data.user_index_to_id[user_index]
        ratings = self.data.get_user_ratings(user_id, "train")
        unrated_indices = np.nonzero(ratings == 0)[0]
        predictions = [
            (item_index, self.predict(user_index, item_index))
            for item_index in unrated_indices
        ]
        predictions = [
            self.data.item_index_to_id[x[0]]
            for x in sorted(predictions, key=lambda x: x[1], reverse=True)
        ]
        return predictions[:n]
