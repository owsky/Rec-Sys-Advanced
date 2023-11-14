from itertools import product
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array
from .CF_Base import CF_Base
from utils import RandomSingleton


class MF(CF_Base):
    """
    Matrix Factorization approach for Collaborative Filtering. Uses sparse arrays and minibatch gradient descent
    """

    def fit(
        self,
        R: coo_array,
        n_factors: int = 10,
        epochs: int = 20,
        lr: float = 0.009,
        reg: float = 0.002,
        batch_size: int = 8,
        lr_decay_factor: float = 0.9,
        max_grad_norm: float | None = 1.0,
    ):
        num_users, num_items = R.shape
        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.max_grad_norm = max_grad_norm
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_set = R

        self.P = RandomSingleton.get_random_normal(
            loc=0, scale=0.1, size=(num_users, n_factors)
        )
        self.Q = RandomSingleton.get_random_normal(
            loc=0, scale=0.1, size=(num_items, n_factors)
        )

        data = list(zip(R.row, R.col, R.data))

        for _ in range(self.epochs):
            self.lr *= self.lr_decay_factor
            RandomSingleton.shuffle(data)

            for i in range(0, len(data), self.batch_size):
                batch = data[i : i + self.batch_size]
                users = np.array([user for user, _, _ in batch])
                items = np.array([item for _, item, _ in batch])
                ratings = np.array([rating for _, _, rating in batch])

                predictions = np.sum(self.P[users, :] * self.Q[items, :], axis=1)
                errors = (ratings - predictions)[:, np.newaxis]

                grad_P = 2 * lr * (errors * self.Q[items, :] - reg * self.P[users, :])
                grad_Q = 2 * lr * (errors * self.P[users, :] - reg * self.Q[items, :])

                self._clip_gradients(grad_P, max_grad_norm)
                self._clip_gradients(grad_Q, max_grad_norm)

                self.P[users, :] += grad_P
                self.Q[items, :] += grad_Q

    def _clip_gradients(
        self, gradient: NDArray[np.float64], max_grad_norm: float | None
    ):
        """
        Avoid overflows in case the gradients diverge during training
        """
        if max_grad_norm is not None:
            norm = np.linalg.norm(gradient)
            if norm > max_grad_norm:
                gradient *= max_grad_norm / norm

    def cross_validate_hyperparameters(
        self,
        train_set: coo_array,
        test_set: coo_array,
        n_factors_range: list[int],
        epochs_range: list[int],
        lr_range: list[float],
        reg_range: list[float],
        batch_size_range: list[int],
        lr_decay_factor_range: list[float],
    ):
        """
        Define the hyperparameter ranges required for crossvalidation, compute the product and invoke the super class' method
        """
        prod = product(
            n_factors_range,
            epochs_range,
            lr_range,
            reg_range,
            batch_size_range,
            lr_decay_factor_range,
        )
        return self._generic_cv_hyper("MF", train_set, test_set, prod)


def cv_hyper_mf_helper(train_set: coo_array, test_set: coo_array):
    print("Grid Search Cross Validation for MF")
    mf = MF()
    n_factors_range = [8, 10, 12]
    epochs_range = [10, 20, 30]
    lr_range = [0.005, 0.009, 0.015]
    reg_range = [0.001, 0.002, 0.003]
    batch_size_range = [2, 4, 8, 16]
    lr_decay_factor_range = [0.5, 0.9, 0.99]
    mf.cross_validate_hyperparameters(
        train_set,
        test_set,
        n_factors_range,
        epochs_range,
        lr_range,
        reg_range,
        batch_size_range,
        lr_decay_factor_range,
    )
