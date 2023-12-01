from abc import ABC, abstractmethod
import math
from typing import Callable
from typing_extensions import Self
from scipy.sparse import coo_array
import numpy as np
from numpy.typing import NDArray


class CF_Base(ABC):
    train_set: coo_array | NDArray[np.float64] | None = None

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
