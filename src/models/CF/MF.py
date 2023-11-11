import math
import numpy as np
from numpy.typing import NDArray
from typing import List, Callable
from scipy.sparse import coo_array
from utils import RandomSingleton


# Matrix Factorization approach for Collaborative Filtering
# Uses sparse arrays and minibatch gradient descent
class MF:
    def __init__(self):
        self.P: NDArray[np.float64] = np.array([])
        self.Q: NDArray[np.float64] = np.array([])

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
        seed: int | None = None,
    ):
        num_users, num_items = R.shape
        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.max_grad_norm = max_grad_norm
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size

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
                errors = ratings - predictions
                self._update_features(
                    errors=errors[:, np.newaxis],
                    user=users,
                    item=items,
                    reg=self.reg,
                    lr=self.lr,
                    max_grad_norm=self.max_grad_norm,
                )

    def predict(self, u: int, i: int) -> float:
        if self.P.size == 0 or self.Q.size == 0:
            raise Exception("Model untrained, invoke fit before predicting")
        return np.dot(self.P[u, :], self.Q[i, :].T)

    # Returns a list containing the results of the loss function per each prediction
    def _compute_prediction_errors(
        self,
        test_set: coo_array,
        error_function: Callable[[float, float], float],
    ) -> List[float]:
        errors = []
        for r, u, i in zip(test_set.data, test_set.row, test_set.col):
            predicted_rating = self.predict(u, i)
            true_rating = float(r)
            errors.append(error_function(true_rating, predicted_rating))
        return errors

    # Returns the Mean Absolute Error accuracy metric for the given test set
    def accuracy_mae(self, user_item_matrix: coo_array) -> float:
        errors = self._compute_prediction_errors(
            user_item_matrix, lambda t, p: abs(t - p)
        )
        mae = float(np.mean(errors))
        return mae

    # Returns the Root Mean Square Error accuracy metric for the given test set
    def accuracy_rmse(self, user_item_matrix: coo_array) -> float:
        errors = self._compute_prediction_errors(
            user_item_matrix, lambda t, p: (t - p) ** 2
        )
        rmse = math.sqrt(np.mean(errors))
        return rmse

    # Avoids overflows in case that gradients diverge during training
    def _clip_gradients(
        self, gradient: NDArray[np.float64], max_grad_norm: float | None
    ):
        if max_grad_norm is not None:
            norm = np.linalg.norm(gradient)
            if norm > max_grad_norm:
                gradient *= max_grad_norm / norm

    def _update_features(
        self,
        errors: float | NDArray[np.float64],
        user: int | NDArray[np.int64],
        item: int | NDArray[np.int64],
        reg: float,
        lr: float,
        max_grad_norm: float | None,
    ):
        grad_P = 2 * lr * (errors * self.Q[item, :] - reg * self.P[user, :])
        grad_Q = 2 * lr * (errors * self.P[user, :] - reg * self.Q[item, :])

        self._clip_gradients(grad_P, max_grad_norm)
        self._clip_gradients(grad_Q, max_grad_norm)

        self.P[user, :] += grad_P
        self.Q[item, :] += grad_Q
