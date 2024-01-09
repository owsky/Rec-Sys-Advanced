from typing import Literal
import numpy as np
from data import Data
import data
from ..CF_Base import CF_Base
from utils import RandomSingleton
from tqdm.auto import tqdm


class ALS(CF_Base):
    """
    Concrete class for Alternating Least Squares recommender system
    """

    def __init__(self, data: Data):
        super().__init__(data, "Alternating Least Squares")

    def fit(self, n_factors=10, epochs=10, reg=0.01, silent=False, cv=False):
        self.is_fit = True
        self.train_set = (
            self.data.interactions_cv_train if cv else self.data.interactions_train
        )
        n_users, n_items = self.train_set.shape

        self.P = RandomSingleton.get_random_normal(
            loc=0, scale=0.1, size=(n_users, n_factors)
        )
        self.Q = RandomSingleton.get_random_normal(
            loc=0, scale=0.1, size=(n_items, n_factors)
        )
        for _ in tqdm(
            range(epochs),
            desc="Fitting the ALS model...",
            leave=False,
            disable=silent,
            dynamic_ncols=True,
        ):
            # Fix item factors and update user factors
            for u in range(n_users):
                # Select the items rated by user u
                observed_items_indices, observed_items = self._get_observed(u, "user")

                self.P[u, :] = np.linalg.solve(
                    (
                        self.Q[observed_items_indices, :].T
                        @ self.Q[observed_items_indices, :]
                    )
                    + reg * np.eye(n_factors),
                    (self.Q[observed_items_indices, :].T @ observed_items),
                )

            # Fix user factors and update item factors
            for i in range(n_items):
                # Select the users who rated item i
                observed_users_indices, observed_users = self._get_observed(i, "item")

                self.Q[i, :] = np.linalg.solve(
                    (
                        self.P[observed_users_indices, :].T
                        @ self.P[observed_users_indices, :]
                    )
                    + reg * np.eye(n_factors),
                    (self.P[observed_users_indices, :].T @ observed_users),
                )
        return self

    def _get_observed(self, index: int, kind: Literal["user", "item"]):
        """
        Return indices and actual values of either a user's or an item's observed ratings
        """
        row, col, data = (self.train_set.row, self.train_set.col, self.train_set.data)
        if kind == "user":
            indices = np.where((row == index))[0]
            sliced_axis = col[indices]
        else:
            indices = np.where((col == index))[0]
            sliced_axis = row[indices]
        sliced_data = data[indices]
        return (sliced_axis, sliced_data)

    def predict(self, u: int, i: int) -> float:
        if not self.is_fit:
            raise RuntimeError("Model untrained, invoke fit before predicting")
        prediction = np.dot(self.P[u, :], self.Q[i, :].T)
        return np.clip(prediction, 1, 5)
