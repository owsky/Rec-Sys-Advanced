from abc import ABC, abstractmethod
import math
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from tqdm import tqdm
from data import Data
from utils.metrics import (
    precision,
    recall,
    f1_score,
    precision_at_k,
    recall_at_k,
    average_reciprocal_hit_rank,
    normalized_discounted_cumulative_gain,
)
from tabulate import tabulate


class Recommender_System(ABC):
    data: Data
    is_fit: bool
    is_biased = False

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

    def accuracy_top_n(self, n=30):
        """
        Compute all accuracy metrics using the test set
        """
        if self.is_fit is None:
            raise RuntimeError("Untrain model, invoke fit before predicting")
        n_users = self.data.interactions_train.shape[0]

        precisions = []
        precisions_at_k = []
        recalls = []
        recalls_at_k = []
        f1_scores = []
        f1_at_k_scores = []
        arhrs = []
        ndcgs = []

        for user_index in tqdm(
            range(n_users), leave=False, desc="Computing accuracy for top N..."
        ):
            test_ratings = csr_array(
                self.data.interactions_test.getrow(user_index)
            ).toarray()[0]
            if self.is_biased:
                user_id = self.data.user_index_to_id[user_index]
                user_bias = self.data.get_user_bias(user_id)
                relevant = np.flatnonzero(test_ratings - user_bias >= 0)
            else:
                relevant = np.flatnonzero(test_ratings >= 3)
            if len(relevant) < 2:
                continue
            elif len(relevant) < n:
                n_adj = len(relevant)
            else:
                n_adj = n
            recommended = [
                self.data.item_id_to_index[x] for x in self.top_n(user_index, n_adj)
            ]
            if len(recommended) < n_adj:
                continue

            prec = precision(relevant, recommended)
            prec_at_k = precision_at_k(relevant, recommended, n_adj)
            rec = recall(relevant, recommended)
            rec_at_k = recall_at_k(relevant, recommended, n_adj)
            arhr = average_reciprocal_hit_rank(relevant, recommended)
            f1 = f1_score(prec, rec)
            f1_at_k = f1_score(prec_at_k, rec_at_k)
            ndcg = normalized_discounted_cumulative_gain(relevant, recommended)
            precisions.append(prec)
            precisions_at_k.append(prec_at_k)
            recalls.append(rec)
            recalls_at_k.append(rec_at_k)
            arhrs.append(arhr)
            f1_scores.append(f1)
            f1_at_k_scores.append(f1_at_k)
            ndcgs.append(ndcg)

        return (
            float(np.mean(precisions)),
            float(np.mean(precisions_at_k)),
            float(np.mean(recalls)),
            float(np.mean(recalls_at_k)),
            float(np.mean(f1_scores)),
            float(np.mean(f1_at_k_scores)),
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
        if not self.is_fit:
            raise RuntimeError("Untrain model, invoke fit before predicting")
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
        print(f"\n{self.model_name} model top N accuracy:")
        table = list(self.accuracy_top_n(n))
        headers = [
            "Precision",
            "Precision@k",
            "Recall",
            "Recall@k",
            "F1",
            "F1@k",
            "Average Reciprocal Hit Rank",
            "Normalized Discounted Cumulative Gain",
        ]
        print(
            tabulate(
                [table],
                headers=headers,
                tablefmt="grid",
                floatfmt=".4f",
                numalign="center",
                stralign="center",
            )
        )

    def pretty_print_accuracy_predictions(self):
        mae = self._accuracy_mae()
        rmse = self._accuracy_rmse()
        print(f"\nModel {self.model_name} predictions accuracy:")
        print(f"MAE: {mae}, RMSE: {rmse}\n")

    @abstractmethod
    def crossvalidation_hyperparameters(self):
        pass
