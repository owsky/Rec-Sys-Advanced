from abc import ABC, abstractmethod
from typing_extensions import Self
from scipy.sparse import coo_array
import numpy as np
from numpy.typing import NDArray

import data
from ..Recommender_System import Recommender_System
from data import Data


class CF_Base(Recommender_System, ABC):
    train_set: coo_array | NDArray[np.float64] | None = None
    data: Data

    @abstractmethod
    def fit(self) -> Self:
        pass

    @abstractmethod
    def predict(self, u: int, i: int) -> float:
        pass

    def _get_cold_start_indices(self, threshold=1):
        """
        Return the indices of the users and items in the train set with less ratings than the threshold
        """
        if self.train_set is None:
            raise RuntimeError("Model untrained, invoke fit before predicting")
        return (
            np.where(np.sum(self.train_set != 0, axis=1) < threshold)[0],
            np.where(np.sum(self.train_set != 0, axis=0) < threshold)[0],
        )

    def _predict_all(self):
        """
        Given a test set, compute the predictions for all the non-zero ratings
        """
        test_set = self.data.test
        predictions: list[tuple[int, int, int, float | None]] = []
        for r, u, i in zip(test_set.data, test_set.row, test_set.col):
            predictions.append((u, i, r, self.predict(u, i)))
        cold_users, cold_items = self._get_cold_start_indices()
        for j, prediction in enumerate(predictions):
            u, i, y_true, _ = prediction
            if u in cold_users or i in cold_items:
                predictions[j] = (u, i, y_true, None)
        return predictions
