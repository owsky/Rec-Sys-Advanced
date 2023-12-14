from numpy import int64
from numpy.typing import NDArray
from data import Data
from ..matrix_factorization import ALS, SGD
from ..neighborhood_based import Neighborhood_Base
import numpy as np
from ..CF_Base import CF_Base


class Ensemble(CF_Base):
    """
    Ensemble of Collaborative Filtering models
    """

    def __init__(
        self, data: Data, sgd_model: SGD, als_model: ALS, nn_model: Neighborhood_Base
    ):
        super().__init__(data, "Ensemble")
        self.train_set = data.interactions_train
        self.ratings_train = data.interactions_train
        self.sgd_model = sgd_model
        self.als_model = als_model
        self.nn_model = nn_model

    def fit(self):
        if not self.sgd_model.is_fit:
            self.sgd_model.fit()
        if not self.als_model.is_fit:
            self.als_model.fit()
        if not self.nn_model.is_fit:
            self.nn_model.fit()
        self.is_fit = True
        return self

    def predict(self, u: int, i: int):
        predictions = []
        predictions.append(self.sgd_model.predict(u, i))
        predictions.append(self.als_model.predict(u, i))
        nn_prediction = self.nn_model.predict(u, i)
        if nn_prediction:
            predictions.append(nn_prediction)
        return float(np.mean(predictions))

    def top_n(self, user_index: int, n: int) -> list[int] | NDArray[int64]:
        if not self.is_fit:
            raise RuntimeError("Untrained model, invoke fit before predicting")
        sgd_recs = self.sgd_model.top_n(user_index, n)
        als_recs = self.als_model.top_n(user_index, n)
        nn_recs = self.nn_model.top_n(user_index, n)

        common = set(sgd_recs) & set(als_recs) & set(nn_recs)

        common_sorted = sorted(
            common,
            key=lambda x: min(sgd_recs.index(x), als_recs.index(x), nn_recs.index(x)),
        )

        remaining = n - len(common_sorted)

        remaining_from_sgd = [item for item in sgd_recs if item not in common][
            : remaining // 3
        ]
        remaining_from_als = [item for item in als_recs if item not in common][
            : remaining // 3
        ]
        remaining_from_nn = [item for item in nn_recs if item not in common][
            : remaining // 3
        ]

        result = (
            common_sorted + remaining_from_sgd + remaining_from_als + remaining_from_nn
        )

        return result

    def gridsearch_cv(self):
        raise RuntimeError(
            f"Model {self.__class__.__name__} has no hyperparameters to crossvalidate"
        )
