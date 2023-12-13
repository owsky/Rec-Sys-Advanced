from itertools import product
from typing import Literal
import numpy as np
from scipy.sparse import coo_array
from data import Data
from ..MF_Base import MF_Base
from utils import RandomSingleton
from typing_extensions import Self


class ALS(MF_Base):
    """
    Concrete class for Alternating Least Squares recommender system
    """

    def __init__(self):
        super().__init__("Alternating Least Squares")

    def fit(self, data: Data, n_factors=10, epochs=10, reg=0.01) -> Self:
        print("Fitting the sequential Alternating Least Squares model...")
        self.data = data
        self.is_fit = True
        n_users, n_items = data.interactions_train.shape

        self.P = RandomSingleton.get_random_normal(
            loc=0, scale=0.1, size=(n_users, n_factors)
        )
        self.Q = RandomSingleton.get_random_normal(
            loc=0, scale=0.1, size=(n_items, n_factors)
        )
        for _ in range(epochs):
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
        row, col, data = (
            self.data.interactions_train.row,
            self.data.interactions_train.col,
            self.data.interactions_train.data,
        )
        if kind == "user":
            indices = np.where((row == index))[0]
            sliced_axis = col[indices]
        else:
            indices = np.where((col == index))[0]
            sliced_axis = row[indices]
        sliced_data = data[indices]
        return (sliced_axis, sliced_data)

    def cross_validate_hyperparameters(
        self,
        train_set: coo_array,
        test_set: coo_array,
        n_factors_range: list[int],
        epochs_range: list[int],
        reg_range: list[float],
    ):
        """
        Define the hyperparameter ranges required for crossvalidation, compute the product and invoke the super class' method
        """
        prod = product(n_factors_range, epochs_range, reg_range)
        return self.crossvalidation_hyperparameters("ALS", train_set, test_set, prod)


def cv_hyper_als_helper(train_set: coo_array, test_set: coo_array):
    print("Grid Search Cross Validation for ALS")
    als = ALS()
    n_factors_range = list(np.linspace(start=2, stop=1000, num=100, dtype=int))
    epochs_range = list(np.linspace(start=10, stop=100, num=20, dtype=int))
    reg_range = list(np.linspace(start=0.001, stop=2.0, num=100, dtype=float))
    als.cross_validate_hyperparameters(
        train_set, test_set, n_factors_range, epochs_range, reg_range
    )
