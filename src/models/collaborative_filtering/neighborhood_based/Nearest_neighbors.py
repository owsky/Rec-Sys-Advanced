from typing import Literal
from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array
from tqdm import tqdm


class Nearest_neighbors:
    def fit(
        self,
        ratings: coo_array,
        kind: Literal["user", "item"],
        similarity: Literal["pearson", "cosine"],
    ):
        rat = ratings.todense()
        dim = rat.shape[0] if kind == "user" else rat.shape[1]
        user_means = (
            None if similarity == "pearson" else np.mean(rat, axis=0, keepdims=True)
        )
        self.similarity_matrix = np.zeros((dim, dim), dtype=np.float64)

        results = Parallel(n_jobs=-1)(
            delayed(self.calculate_similarity)(
                rat, user_means, i, dim, kind, similarity
            )
            for i in tqdm(range(dim))
        )

        for index, result in enumerate(results):
            self.similarity_matrix[index, :] = result

    def calculate_similarity(
        self,
        rat,
        user_means,
        i,
        dim,
        kind: Literal["user", "item"],
        similarity: Literal["pearson", "cosine"],
    ):
        result_row = np.zeros(dim)
        for j in range(dim):
            if similarity == "pearson":
                result_row[j] = self.pearson_correlation(rat, i, j, kind)
            else:
                result_row[j] = self.adjusted_cosine_similarity(
                    rat, user_means, i, j, kind
                )
        return result_row

    def _get_common_ratings(
        self, ratings: NDArray, i: int, j: int, kind: Literal["user", "item"]
    ):
        # Extract the total user or item ratings
        if kind == "user":
            ratings_i = ratings[i, :]
            ratings_j = ratings[j, :]
        else:
            ratings_i = ratings[:, i]
            ratings_j = ratings[:, j]

        # Find the indices where both users have non-zero ratings
        common_ratings_mask = (ratings_i != 0) & (ratings_j != 0)

        # If there are no common ratings, return 0 (no correlation)
        if np.sum(common_ratings_mask) == 0:
            return ()

        # Select only the common ratings for both users
        common_ratings_i = ratings_i[common_ratings_mask]
        common_ratings_j = ratings_j[common_ratings_mask]

        # Avoid division by zero in case of zero variance
        if np.var(common_ratings_i) == 0 or np.var(common_ratings_j) == 0:
            return ()
        return common_ratings_i, common_ratings_j

    def pearson_correlation(
        self, ratings: NDArray, i: int, j: int, kind: Literal["user", "item"]
    ) -> float:
        common_ratings = self._get_common_ratings(ratings, i, j, kind)
        if len(common_ratings) == 0:
            return 0.0
        else:
            common_ratings_i, common_ratings_j = common_ratings

        # Compute the Pearson correlation coefficient
        correlation_coefficient = np.corrcoef(common_ratings_i, common_ratings_j)[0, 1]

        # If there are NaN values in the correlation coefficient, return 0 (no correlation)
        if np.isnan(correlation_coefficient):
            return 0.0

        return correlation_coefficient

    def adjusted_cosine_similarity(
        self,
        ratings: NDArray,
        user_means: NDArray,
        i: int,
        j: int,
        kind: Literal["user", "item"],
    ) -> float:
        adjusted_ratings = ratings - user_means
        common_ratings = self._get_common_ratings(adjusted_ratings, i, j, kind)
        if len(common_ratings) == 0:
            return 0.0
        else:
            common_ratings_i, common_ratings_j = common_ratings

        # Compute the cosine similarity
        similarity = np.dot(common_ratings_i, common_ratings_j) / (
            float(np.linalg.norm(common_ratings_i))
            * float(np.linalg.norm(common_ratings_j))
        )

        # If there are NaN values in the similarity, return 0 (no similarity)
        if np.isnan(similarity):
            return 0.0

        return similarity

    def get_recommendations(self, i: int, k: int):
        similarity_scores = self.similarity_matrix[i, :]
        sorted_indices = np.argsort(similarity_scores)[::-1]
        return sorted_indices[1 : k + 1]
