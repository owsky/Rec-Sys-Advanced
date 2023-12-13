import numpy as np
from typing import Literal
from .Neighborhood_Base import Neighborhood_Base


class Item_Based(Neighborhood_Base):
    def __init__(
        self, kind: Literal["user", "item"], similarity: Literal["pearson", "cosine"]
    ):
        super().__init__("Item-based Neighborhood Filtering")
        self.kind = kind
        self.similarity = similarity

    def predict(self, u: int, i: int, k=50, support=3):
        # Extract the top k neighbors of the item i which have been rated by user u
        similarities = self.similarity_matrix[i, :]
        top_neighbors = np.argsort(similarities)[::-1]
        valid_neighbors = top_neighbors[
            (self.data.interactions_train_numpy[u, top_neighbors] != 0)
            & (self.similarity_matrix[i, top_neighbors] > 0)
        ][:k]

        # Skip if support is not met, i.e., ^r(u,i)=0
        if len(valid_neighbors) < support:
            return None

        # Biases required for rating normalization
        bias_i = self.data.average_item_rating[i]
        bias_j = self.data.average_item_rating[valid_neighbors]

        # Compute the prediction as the adjusted ratings times the similarity score
        # divided by the sum of the similarities
        num = np.sum(
            self.similarity_matrix[i, valid_neighbors]
            * (self.data.interactions_train_numpy[u, valid_neighbors] - bias_j)
        )
        den = np.sum(self.similarity_matrix[i, valid_neighbors])

        # Avoid division by zero
        if den != 0:
            return bias_i + float(num) / float(den)
