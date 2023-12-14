import numpy as np
from numpy.typing import NDArray
from data import Data
from ..MF_Base import MF_Base
from utils import RandomSingleton
from typing_extensions import Self
from tqdm import tqdm


class SGD(MF_Base):
    """
    Matrix Factorization model which uses Stochastic Gradient Descent for training
    """

    def __init__(self, data: Data):
        super().__init__(data, "Stochastic Gradient Descent")

    def fit(
        self,
        n_factors: int = 5,
        epochs: int = 10,
        lr: float = 0.011,
        reg: float = 0.002,
        batch_size: int = 8,
        lr_decay_factor: float = 0.5,
        silent=False,
    ) -> Self:
        """
        Mini batch SGD training algorithm
        """
        self.is_fit = True
        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        num_users, num_items = self.data.interactions_train.shape

        self.P = RandomSingleton.get_random_normal(
            loc=0, scale=0.1, size=(num_users, n_factors)
        )
        self.Q = RandomSingleton.get_random_normal(
            loc=0, scale=0.1, size=(num_items, n_factors)
        )

        iterable_data = list(
            zip(
                self.data.interactions_train.row,
                self.data.interactions_train.col,
                self.data.interactions_train.data,
            )
        )

        for _ in tqdm(
            range(self.epochs),
            leave=False,
            desc="Fitting the Stochastic Gradient Descent model...",
            disable=silent,
            dynamic_ncols=True,
        ):
            self.lr *= self.lr_decay_factor
            RandomSingleton.shuffle(iterable_data)

            for i in range(0, len(iterable_data), self.batch_size):
                batch = iterable_data[i : i + self.batch_size]
                users = np.array([user for user, _, _ in batch])
                items = np.array([item for _, item, _ in batch])
                ratings = np.array([rating for _, _, rating in batch])

                predictions = np.sum(self.P[users, :] * self.Q[items, :], axis=1)
                errors = (ratings - predictions)[:, np.newaxis]

                grad_P = 2 * lr * (errors * self.Q[items, :] - reg * self.P[users, :])
                grad_Q = 2 * lr * (errors * self.P[users, :] - reg * self.Q[items, :])

                self._clip_gradients(grad_P)
                self._clip_gradients(grad_Q)

                self.P[users, :] += grad_P
                self.Q[items, :] += grad_Q
        return self

    def _clip_gradients(self, gradient: NDArray[np.float64]):
        """
        Clip gradients in order to make sure that they don't diverge
        """
        norm = float(np.linalg.norm(gradient))
        if norm > 1.0:
            gradient /= norm
