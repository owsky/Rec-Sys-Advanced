import math
from typing import Callable
from data import Data
from ..matrix_factorization import ALS, SVD
from ..neighborhood_based import Neighborhood_Based
from scipy.sparse import coo_array
import numpy as np


class Ensemble:
    def __init__(
        self, data: Data, svd_model: SVD, als_model: ALS, nn_model: Neighborhood_Based
    ):
        self.train_set = data.train
        self.svd_model = (
            svd_model.fit(data.train) if svd_model.train_set is None else svd_model
        )
        self.als_model = (
            als_model.fit(data.train) if als_model.train_set is None else als_model
        )
        self.nn_model = nn_model.fit(data) if nn_model.train_set is None else nn_model

    def predict(self, u: int, i: int):
        predictions = []
        predictions.append(self.svd_model.predict(u, i))
        predictions.append(self.als_model.predict(u, i))
        nn_prediction = self.nn_model.predict(u, i)
        if nn_prediction:
            predictions.append(nn_prediction)
        return float(np.mean(predictions))

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
