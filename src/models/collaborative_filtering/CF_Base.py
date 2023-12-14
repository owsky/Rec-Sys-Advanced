from abc import ABC
import numpy as np
from ..Recommender_System import Recommender_System


class CF_Base(Recommender_System, ABC):
    def _get_cold_start_indices(self, threshold=1):
        """
        Return the indices of the users and items in the train set with less ratings than the threshold
        """
        return (
            np.where(np.sum(self.data.interactions_train != 0, axis=1) < threshold)[0],
            np.where(np.sum(self.data.interactions_train != 0, axis=0) < threshold)[0],
        )

    def _predict_all(self) -> list[tuple[int, int, float, float]]:
        """
        Compute the predictions for all the non-zero ratings
        """
        test_set = self.data.interactions_test
        cold_users, cold_items = self._get_cold_start_indices()
        y = zip(test_set.data, test_set.row, test_set.col)
        predictions = [
            (u, i, r, self.predict(u, i))
            for r, u, i in y
            if u not in cold_users and i not in cold_items
        ]
        return predictions
