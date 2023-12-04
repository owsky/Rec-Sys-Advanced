from typing_extensions import Self
from numpy import int64
from numpy.typing import NDArray
from data import Data
from ..matrix_factorization import ALS, SVD
from ..neighborhood_based import Neighborhood_Base
import numpy as np
from ..CF_Base import CF_Base


class Ensemble(CF_Base):
    def __init__(
        self, data: Data, svd_model: SVD, als_model: ALS, nn_model: Neighborhood_Base
    ):
        super().__init__("Ensemble")
        self.data = data
        self.train_set = data.train
        self.ratings_train = data.train
        self.svd_model = svd_model
        self.als_model = als_model
        self.nn_model = nn_model

    def fit(self) -> Self:
        if self.svd_model.train_set is None:
            self.svd_model.fit(self.data)
        if self.als_model.train_set is None:
            self.als_model.fit(self.data)
        if self.nn_model.train_set is None:
            self.nn_model.fit(self.data)
        return self

    def predict(self, u: int, i: int):
        predictions = []
        predictions.append(self.svd_model.predict(u, i))
        predictions.append(self.als_model.predict(u, i))
        nn_prediction = self.nn_model.predict(u, i)
        if nn_prediction:
            predictions.append(nn_prediction)
        return float(np.mean(predictions))

    def top_n(self, user_index: int, n: int) -> list[int] | NDArray[int64]:
        return super().top_n(user_index, n)

    def crossvalidation_hyperparameters(self):
        raise RuntimeError(
            f"Model {self.__class__.__name__} has no hyperparameters to crossvalidate"
        )
