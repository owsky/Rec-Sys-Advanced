from abc import ABC, abstractmethod
import math
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array, csc_array, csr_array
from tqdm import tqdm
from data import Data
from utils.metrics import (
    f1_score,
    precision_at_k,
    recall_at_k,
    average_reciprocal_hit_rank,
    normalized_discounted_cumulative_gain,
)


class Recommender_System(ABC):
    ratings_train: coo_array | csc_array | csr_array | NDArray[np.float64] | None
    data: Data

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def top_n(self, user_index: int, n: int) -> list[int] | NDArray[np.int64]:
        pass

    def accuracy_top_n(self, n=30) -> tuple[float, float, float, float, float]:
        """
        Compute all accuracy metrics using the test set
        """
        if self.ratings_train is None:
            raise RuntimeError("Untrain model, invoke fit before predicting")
        n_users = self.ratings_train.shape[0]

        precisions = []
        recalls = []
        f1_scores = []
        arhrs = []
        ndcgs = []
        for user_index in tqdm(
            range(n_users), leave=False, desc="Computing accuracy for top N..."
        ):
            test_ratings = csr_array(self.data.test.getrow(user_index)).toarray()[0]
            relevant = np.flatnonzero(test_ratings >= 3)
            recommended = [
                self.data.item_id_to_index[x] for x in self.top_n(user_index, n)
            ]

            if len(relevant) >= 1 and len(recommended) >= 1:
                precision = precision_at_k(relevant, recommended, n)
                recall = recall_at_k(relevant, recommended, n)
                arhr = average_reciprocal_hit_rank(relevant, recommended)
                f1 = f1_score(precision, recall)
                ndcg = normalized_discounted_cumulative_gain(relevant, recommended)
                precisions.append(precision)
                recalls.append(recall)
                arhrs.append(arhr)
                f1_scores.append(f1)
                ndcgs.append(ndcg)

        return (
            float(np.mean(precisions)),
            float(np.mean(recalls)),
            float(np.mean(f1_scores)),
            float(np.mean(arhrs)),
            float(np.mean(ndcgs)),
        )

    @abstractmethod
    def _predict_all(self) -> list[tuple[int, int, int, float | None]]:
        pass

    def _compute_prediction_errors(self, loss_function: Callable):
        """
        Given a test set and a loss function, compute predictions and errors
        """
        predictions = self._predict_all()
        errors = []
        for prediction in predictions:
            _, _, y_true, y_pred = prediction
            if y_pred is not None:
                errors.append(loss_function(y_true - y_pred))
        return errors

    def _accuracy_mae(self):
        """
        Compute the Mean Absolute Error of the trained model on a test set
        """
        errors = self._compute_prediction_errors(lambda x: abs(x))
        return np.mean(errors)

    def _accuracy_rmse(self):
        """
        Compute the Root Mean Squared Error of the trained model on a test set
        """
        errors = self._compute_prediction_errors(lambda x: x**2)
        return math.sqrt(np.mean(errors))

    def pretty_print_accuracy_top_n(self, n=30):
        precision, recall, f1, arhr, ndcg = self.accuracy_top_n(n)
        print(f"\n{self.model_name} model top N accuracy:")
        print(
            f"Precision@k: {precision:.3f}, Recall@k: {recall:.3f}, ",
            f"F1: {f1:.3f}, Average Reciprocal Hit Rank: {arhr:.3f}, ",
            f"Normalized Discounted Cumulative Gain: {ndcg:.3f}\n",
        )

    def pretty_print_accuracy_predictions(self):
        mae = self._accuracy_mae()
        rmse = self._accuracy_rmse()
        print(f"\nModel {self.model_name} predictions accuracy:")
        print(f"MAE: {mae}, RMSE: {rmse}\n")

    @abstractmethod
    def crossvalidation_hyperparameters(self):
        pass
