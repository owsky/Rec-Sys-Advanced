from abc import ABC
import numpy as np
from ..Recommender_System import Recommender_System
from scipy.sparse import coo_array


class CF_Base(Recommender_System, ABC):
    def _get_cold_start_indices(self, threshold=1):
        """
        Return the indices of the users and items in the train set with less ratings than the threshold
        """
        return (
            np.where(np.sum(self.data.interactions_train != 0, axis=1) < threshold)[0],
            np.where(np.sum(self.data.interactions_train != 0, axis=0) < threshold)[0],
        )

    def _predict_all(self, test_set: coo_array) -> list[tuple[int, int, float, float]]:
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

    def top_n(self, user_index: int, n=10, cv=False):
        if not self.is_fit:
            raise RuntimeError("Model untrained, fit first")
        user_id = self.data.user_index_to_id[user_index]
        ratings = self.data.get_user_ratings(user_id, "train_cv" if cv else "train")
        unrated_indices = np.flatnonzero(ratings == 0)
        predictions = [
            (item_index, self.predict(user_index, item_index))
            for item_index in unrated_indices
            if self.predict(user_index, item_index) is not None
        ]
        predictions = [
            self.data.item_index_to_id[x[0]]
            for x in sorted(predictions, key=lambda x: x[1], reverse=True)
        ]
        return predictions[:n]
