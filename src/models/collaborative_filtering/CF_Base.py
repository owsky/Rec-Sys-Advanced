from abc import ABC, abstractmethod
from itertools import product
import math
from typing import Callable
from scipy.sparse import coo_array
import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate
from cross_validation import grid_search


class CF_Base(ABC):
    """
    Base class for Collaborative Filtering recommender systems
    """

    def __init__(self):
        self.P: NDArray[np.float64] = np.array([])
        self.Q: NDArray[np.float64] = np.array([])
        self.train_set: coo_array | None = None

    @abstractmethod
    def fit(self, R: coo_array):
        pass

    def predict(self, u: int, i: int) -> float:
        if self.P.size == 0 or self.Q.size == 0:
            raise RuntimeError("Model untrained, invoke fit before predicting")
        return np.dot(self.P[u, :], self.Q[i, :].T)

    def _get_cold_start_indices(self, threshold=0):
        """
        Return the indices of the users and items in the train set with less ratings than the threshold
        """
        if self.train_set is None:
            raise RuntimeError("Model untrained, invoke fit before predicting")
        return (
            np.where(np.sum(self.train_set != 0, axis=1) < threshold)[0],
            np.where(np.sum(self.train_set != 0, axis=0) < threshold)[0],
        )

    def predict_all(self, test_set: coo_array):
        """
        Given a test set, compute the predictions for all the non-zero ratings
        """
        predictions: list[tuple[int, int, int, float | None]] = []
        for r, u, i in zip(test_set.data, test_set.row, test_set.col):
            predictions.append((u, i, r, self.predict(u, i)))
        cold_users, cold_items = self._get_cold_start_indices()
        for j, prediction in enumerate(predictions):
            u, i, y_true, _ = prediction
            if u in cold_users or i in cold_items:
                predictions[j] = (u, i, y_true, None)
        return predictions

    def _compute_prediction_errors(self, test_set: coo_array, loss_function: Callable):
        """
        Given a test set and a loss function, compute predictions and errors
        """
        predictions = self.predict_all(test_set)
        errors = []
        for prediction in predictions:
            _, _, y_true, y_pred = prediction
            if y_pred is not None:
                errors.append(loss_function(y_true - y_pred))
        return errors

    def accuracy_mae(self, test_set: coo_array):
        """
        Compute the Mean Absolute Error of the trained model on a test set
        """
        errors = self._compute_prediction_errors(test_set, lambda x: abs(x))
        return np.mean(errors)

    def accuracy_rmse(self, test_set: coo_array):
        """
        Compute the Root Mean Squared Error of the trained model on a test set
        """
        errors = self._compute_prediction_errors(test_set, lambda x: x**2)
        return math.sqrt(np.mean(errors))

    def _generic_cv_hyper(
        self,
        model_label: str,
        train_set: coo_array,
        test_set: coo_array,
        prod: product,
        sequential=False,
    ):
        """
        Helper function invoked by the concrete classes to compute and output the results of hyperparameter crossvalidation
        """
        results = grid_search(self, train_set, test_set, prod, sequential)
        results = sorted(results, key=lambda x: (x[1], x[2]))
        table = []
        best_params = ()
        for i, result in enumerate(results):
            if i == 0:
                best_params = result[0]
            table.append(list(result))
        headers = ["Hyperparameters", "MAE", "RMSE"]
        print(f"Crossvalidation results for {model_label}")
        print(tabulate(table, headers=headers, tablefmt="grid"))
        return best_params
