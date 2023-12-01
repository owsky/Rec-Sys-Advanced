from abc import ABC, abstractmethod
from itertools import product
from scipy.sparse import coo_array
import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate
from cross_validation import grid_search
from ..CF_Base import CF_Base


class MF_Base(CF_Base, ABC):
    """
    Base class for Collaborative Filtering recommender systems
    """

    train_set: coo_array | None = None

    def __init__(self):
        self.P: NDArray[np.float64] = np.array([])
        self.Q: NDArray[np.float64] = np.array([])

    @abstractmethod
    def fit(self, R: coo_array) -> "MF_Base":
        pass

    def predict(self, u: int, i: int) -> float:
        if self.P.size == 0 or self.Q.size == 0:
            raise RuntimeError("Model untrained, invoke fit before predicting")
        return np.dot(self.P[u, :], self.Q[i, :].T)

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
        results = sorted(results, key=lambda x: (x[1], x[2]))[:10]
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
