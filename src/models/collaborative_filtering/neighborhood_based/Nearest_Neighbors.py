from math import sqrt
from typing import Callable, Literal
from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from data import Data
from scipy.stats import pearsonr


class Nearest_Neighbors:
    errors: list[float] | None = None

    def fit(
        self,
        data: Data,
        kind: Literal["user", "item"],
        similarity: Literal["pearson", "cosine"],
    ):
        """
        Compute the similarity matrix for the input data using either Pearson Correlation
        or Adjusted Cosine Similarity. Parameter kind determines whether user-user or item-item strategy
        for recommendations is adopted.
        """
        print(
            f"Fitting the {kind}-based Neighborhood Filtering model with {similarity}..."
        )
        self.data = data
        self.ratings = data.train.todense().astype(np.float64)
        self.test = data.test.todense().astype(np.float64)
        self.kind = kind
        self.similarity = similarity
        if kind == "user":
            dim = data.train.shape[0]
        elif kind == "item":
            dim = data.train.shape[1]
        else:
            raise RuntimeError("Wrong value for parameter kind")

        self.similarity_matrix = np.zeros((dim, dim), dtype=np.float64)

        # Compute the similarity matrix in parallel
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.calculate_similarity)(i, dim)
            for i in tqdm(
                range(dim),
                desc="Computing " + kind + "-based similarities using " + similarity,
            )
        )

        # Store the results into the similarity matrix
        for index, result in enumerate(results):
            self.similarity_matrix[index, :] = result

        # Copy the top half of the similarity matrix into the bottom half, alongside the main diagonal
        rows, cols = self.similarity_matrix.shape
        for i in range(rows):
            for j in range(i + 1, cols):
                self.similarity_matrix[j, i] = self.similarity_matrix[i, j]

    def calculate_similarity(self, i: int, dim: int) -> NDArray[np.float64]:
        """
        Parallelized inner loop for computing the similarity matrix. Only computes pairwise similarities
        above the main diagonal for efficiency reasons.
        """
        result_row = np.zeros(dim)
        for j in range(i + 1, dim):
            if self.similarity == "pearson":
                result_row[j] = self.pearson_correlation(i, j)
            else:
                result_row[j] = self.adjusted_cosine_similarity(i, j)
        return result_row

    def _get_common_ratings(
        self, i: int, j: int
    ) -> tuple[()] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Extract either all the users who have rated both movies i and j
        or all the items rated by both users i and j through their respective ratings
        """
        # Extract the total user or item ratings
        if self.kind == "user":
            ratings_i = self.ratings[i, :]
            ratings_j = self.ratings[j, :]
        else:
            ratings_i = self.ratings[:, i]
            ratings_j = self.ratings[:, j]

        # Find the indices where both users have non-zero ratings
        common_ratings_mask = (ratings_i != 0) & (ratings_j != 0)

        # If there are no common ratings, return empty tuple for no correlation
        if np.sum(common_ratings_mask) == 0:
            return ()

        # Select only the common ratings for both users
        common_ratings_i = ratings_i[common_ratings_mask]
        common_ratings_j = ratings_j[common_ratings_mask]

        # Avoid division by zero in case of zero variance
        if np.var(common_ratings_i) == 0 or np.var(common_ratings_j) == 0:
            return ()
        return common_ratings_i, common_ratings_j

    def pearson_correlation(self, i: int, j: int) -> float:
        """
        Compute the Pearson Correlation between either the items i and j or ther users i and j
        """

        # Extract the common ratings between i and j
        common_ratings = self._get_common_ratings(i, j)
        if len(common_ratings) == 0:
            return 0.0

        common_ratings_i, common_ratings_j = common_ratings

        return pearsonr(
            common_ratings_i - np.mean(common_ratings_i),
            common_ratings_j - np.mean(common_ratings_j),
        ).correlation

    def adjusted_cosine_similarity(self, i: int, j: int) -> float:
        """
        Compute the Adjusted Cosine Similarity between either items i and j or users i and j
        """

        # Extract the common ratings between i and j
        common_ratings = self._get_common_ratings(i, j)
        if len(common_ratings) == 0:
            return 0.0

        common_ratings_i, common_ratings_j = common_ratings

        # Compute the user or item biases
        if self.kind == "user":
            bias_i = self.data.average_user_rating[i]
            bias_j = self.data.average_user_rating[j]
        else:
            bias_i = self.data.average_item_rating[i]
            bias_j = self.data.average_item_rating[j]

        # Compute the adjusted cosine similarity
        num = float(np.dot(common_ratings_i - bias_i, common_ratings_j - bias_j))
        den_i = float(np.linalg.norm(common_ratings_i - bias_i))
        den_j = float(np.linalg.norm(common_ratings_j - bias_j))
        den = den_i * den_j

        # Avoid division by zero
        if den == 0:
            return 0.0

        return num / den

    def predict_all(self, k=50, support=3):
        """
        Compute predictions for each non-zero rating in the test set.
        k is the number of neighbors to consider and support is the minimum number
        of neighbors required for computing the predictions.
        """
        predicted_ratings = np.zeros_like(self.test)
        # Non-zero ratings' indices
        nz_indices = zip(*self.test.nonzero())

        for u, i in nz_indices:
            # Item-based model
            if self.kind == "item":
                # Extract the top k neighbors of the item i which have been rated by user u
                similarities = self.similarity_matrix[i, :]
                top_neighbors = np.argsort(similarities)[::-1]
                valid_neighbors = top_neighbors[
                    (self.ratings[u, top_neighbors] != 0)
                    & (self.similarity_matrix[i, top_neighbors] > 0)
                ][:k]

                # Skip if support is not met, i.e., ^r(u,i)=0
                if len(valid_neighbors) < support:
                    continue

                # Biases required for rating normalization
                bias_i = self.data.average_item_rating[i]
                bias_j = self.data.average_item_rating[valid_neighbors]

                # Compute the prediction as the adjusted ratings times the similarity score
                # divided by the sum of the similarities
                num = np.sum(
                    self.similarity_matrix[i, valid_neighbors]
                    * (self.ratings[u, valid_neighbors] - bias_j)
                )
                den = np.sum(self.similarity_matrix[i, valid_neighbors])

                # Avoid division by zero
                if den != 0:
                    predicted_ratings[u, i] = bias_i + float(num) / float(den)
            else:
                # Extract the top k neighbors of the user u who have rated the item i
                similarities = self.similarity_matrix[u, :]
                top_neighbors = np.argsort(similarities)[::-1]
                valid_neighbors = top_neighbors[
                    (self.ratings[top_neighbors, i] != 0)
                    & (self.similarity_matrix[u, top_neighbors] > 0)
                ][:k]

                # Skip if support is not met, i.e., ^r(u,i)=0
                if len(valid_neighbors) < support:
                    continue

                # Biases required for rating normalization
                bias_u = self.data.average_user_rating[u]
                bias_v = self.data.average_user_rating[valid_neighbors]

                # Compute the prediction as the adjusted ratings times the similarity score
                # divided by the sum of the similarities
                num = np.sum(
                    self.similarity_matrix[u, valid_neighbors]
                    * (self.ratings[valid_neighbors, i] - bias_v)
                )
                den = np.sum(self.similarity_matrix[u, valid_neighbors])

                # Avoid division by zero
                if den != 0:
                    predicted_ratings[u, i] = bias_u + float(num) / float(den)

        return predicted_ratings

    def _compute_errors(self, loss: Callable) -> list[float]:
        predictions = self.predict_all()
        n_users, n_items = self.test.shape

        errors = [
            loss(predictions[u, i] - self.test[u, i])
            for u in range(n_users)
            for i in range(n_items)
            if self.test[u, i] != 0 and predictions[u, i] != 0
        ]
        return errors

    def accuracy_rmse(self) -> float:
        """
        Compute the Root Mean Squared Error on the test set
        """
        if self.errors is None:
            self.errors = self._compute_errors(lambda x: x**2)

        return sqrt(np.mean(self.errors))

    def accuracy_mae(self) -> float:
        """
        Compute the Mean Absolute Error on the test set
        """
        if self.errors is None:
            self.errors = self._compute_errors(lambda x: abs(x))

        return float(np.mean(self.errors))
