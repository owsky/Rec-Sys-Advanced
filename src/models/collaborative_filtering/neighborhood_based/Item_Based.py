import numpy as np
from typing import Literal
from data import Data
from .Neighborhood_Base import Neighborhood_Base


class Item_Based(Neighborhood_Base):
    def __init__(
        self,
        data: Data,
        similarity: Literal["pearson", "cosine"],
    ):
        super().__init__(data, "Item-based Neighborhood Filtering", "item", similarity)

    def predict(self, u: int, i: int, k=50, support=3) -> float | None:
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

    def top_n(self, user_index: int, n=10) -> list[int]:
        if not self.is_fit:
            raise RuntimeError("Model untrained, fit first")
        user_id = self.data.user_index_to_id[user_index]
        ratings = self.data.get_user_ratings(user_id, "train")
        unrated_indices = np.flatnonzero(ratings == 0)
        predictions: list[tuple[int, float]] = []
        for item_index in unrated_indices:
            pred = self.predict(user_index, item_index)
            if pred is not None:
                predictions.append((item_index, pred))
        return [
            self.data.item_index_to_id[index]
            for (index, _) in sorted(predictions, key=lambda x: x[1], reverse=True)
        ][:n]
