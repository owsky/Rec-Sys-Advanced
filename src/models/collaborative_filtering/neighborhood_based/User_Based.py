from typing import Literal
import numpy as np
from data import Data
from .Neighborhood_Base import Neighborhood_Base


class User_Based(Neighborhood_Base):
    def __init__(
        self,
        data: Data,
        kind: Literal["user", "item"],
        similarity: Literal["pearson", "cosine"],
    ):
        super().__init__(data, "User-based Neighborhood Filtering")
        self.kind = kind
        self.similarity = similarity

    def predict(self, u: int, i: int, k=50, support=3):
        # Extract the top k neighbors of the user u who have rated the item i
        similarities = self.similarity_matrix[u, :]
        top_neighbors = np.argsort(similarities)[::-1]
        valid_neighbors = top_neighbors[
            (self.data.interactions_train_numpy[top_neighbors, i] != 0)
            & (self.similarity_matrix[u, top_neighbors] > 0)
        ][:k]

        # Skip if support is not met, i.e., ^r(u,i)=0
        if len(valid_neighbors) < support:
            return None

        # Biases required for rating normalization
        bias_u = self.data.average_user_rating[u]
        bias_v = self.data.average_user_rating[valid_neighbors]

        # Compute the prediction as the adjusted ratings times the similarity score
        # divided by the sum of the similarities
        num = np.sum(
            self.similarity_matrix[u, valid_neighbors]
            * (self.data.interactions_train_numpy[valid_neighbors, i] - bias_v)
        )
        den = np.sum(self.similarity_matrix[u, valid_neighbors])

        # Avoid division by zero
        if den != 0:
            return bias_u + float(num) / float(den)
