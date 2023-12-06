from typing import Literal
from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from data import Data
from scipy.stats import pearsonr
from ..CF_Base import CF_Base
from typing_extensions import Self


class Neighborhood_Base(CF_Base):
    ratings_train: NDArray[np.float64] | None
    kind: Literal["user", "item"]
    similarity: Literal["pearson", "cosine"]

    def fit(self, data: Data) -> Self:
        """
        Compute the similarity matrix for the input data using either Pearson Correlation
        or Adjusted Cosine Similarity. Parameter kind determines whether user-user or item-item strategy
        for recommendations is adopted.
        """
        print(
            f"Fitting the {self.kind}-based Neighborhood Filtering model with {self.similarity}..."
        )
        self.data = data
        self.ratings = data.train.todense().astype(np.float64)
        self.ratings_train = self.ratings

        if self.kind == "user":
            dim = data.train.shape[0]
        elif self.kind == "item":
            dim = data.train.shape[1]
        else:
            raise RuntimeError("Wrong value for parameter kind")

        self.similarity_matrix = np.zeros((dim, dim), dtype=np.float64)

        # Compute the similarity matrix in parallel
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.calculate_similarity)(i, dim)
            for i in tqdm(
                range(dim),
                desc="Computing "
                + self.kind
                + "-based similarities using "
                + self.similarity,
                leave=False,
            )
        )

        # Store the results into the similarity matrix
        for index, result in enumerate(results):
            self.similarity_matrix[index, :] = result

        # Copy the top half of the similarity matrix into the bottom half, alongside the main diagonal
        self.similarity_matrix += self.similarity_matrix.T
        return self

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

    def top_n(self, user_index: int, n=10):
        if self.ratings_train is None:
            raise RuntimeError("Model untrained, fit first")
        ratings = self.ratings_train[user_index]
        unrated_indices = np.nonzero(ratings == 0)[0]
        predictions = [
            (item_index, self.predict(user_index, item_index))
            for item_index in unrated_indices
        ]
        predictions = [x for x in predictions if x[1] is not None]
        predictions = [
            self.data.item_index_to_id[x[0]]
            for x in sorted(predictions, key=lambda x: x[1], reverse=True)
        ]
        return predictions[:n]

    def crossvalidation_hyperparameters(self):
        raise RuntimeError(
            f"Model {self.__class__.__name__} has no hyperparameters to crossvalidate"
        )
