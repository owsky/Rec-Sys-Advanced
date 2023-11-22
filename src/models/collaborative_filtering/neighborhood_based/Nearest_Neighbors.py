from math import sqrt
from typing import Literal
from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from tqdm import tqdm
from data import Data


class Nearest_Neighbors:
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
        self.data = data
        self.ratings = data.ratings.todense().astype(np.float64)
        self.adj_ratings = self.ratings.copy()
        self.kind = kind
        self.similarity = similarity
        if kind == "user":
            dim = data.ratings.shape[0]
        elif kind == "item":
            dim = data.ratings.shape[1]
        else:
            raise RuntimeError("Wrong value for argument kind")
        self.similarity_matrix = np.zeros((dim, dim), dtype=np.float64)

        # Precompute centering of ratings by subtracting the mean of the users' ratings
        if similarity == "cosine":
            rows, cols = self.ratings.shape
            for i in range(rows):
                for j in range(cols):
                    if self.ratings[i, j] != 0:
                        self.adj_ratings[i, j] -= data.average_user_rating[i]
        elif similarity == "pearson":
            rows, cols = self.ratings.shape
            for i in range(rows):
                for j in range(cols):
                    if self.ratings[i, j] != 0:
                        self.adj_ratings[i, j] -= data.average_item_rating[j]
        else:
            raise RuntimeError(f"{similarity} similarity metric not implement")

        # Compute the similarity matrix in parallel
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.calculate_similarity)(i, dim)
            for i in tqdm(range(dim), desc="Computing similarities")
        )

        # Store the results into the similarity matrix
        for index, result in enumerate(results):
            self.similarity_matrix[index, :] = result

        # Copy the top half of the similarity matrix into the bottom half, alongside
        # the main diagonal
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
            ratings_i = self.adj_ratings[i, :]
            ratings_j = self.adj_ratings[j, :]
        else:
            ratings_i = self.adj_ratings[:, i]
            ratings_j = self.adj_ratings[:, j]

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

    # TODO try to normalize the correlation by the number of common users/items
    def pearson_correlation(self, i: int, j: int) -> float:
        """
        Compute the Pearson Correlation between either items i and j or users i and j
        """

        # Extract the common ratings between i and j
        common_ratings = self._get_common_ratings(i, j)
        if len(common_ratings) == 0:
            return 0.0

        common_ratings_i, common_ratings_j = common_ratings

        num = float(np.dot(common_ratings_i, common_ratings_j))
        den_i = float(np.sum(common_ratings_i**2))
        den_j = float(np.sum(common_ratings_j**2))
        den = sqrt(den_i * den_j)
        if den == 0:
            return 0.0
        return num / den

    def adjusted_cosine_similarity(self, i: int, j: int) -> float:
        """
        Compute the Adjusted Cosine Similarity between either items i and j or users i and j
        """

        # Extract the common ratings
        common_ratings = self._get_common_ratings(i, j)

        # If there are no common ratings then return zero for non correlation
        if len(common_ratings) == 0:
            return 0.0

        # Compute the cosine similarity
        common_ratings_i, common_ratings_j = common_ratings

        num = float(np.dot(common_ratings_i, common_ratings_j))
        den = float(np.linalg.norm(common_ratings_i, ord=2)) * float(
            np.linalg.norm(common_ratings_j, ord=2)
        )

        if den == 0:
            return 0.0  # Avoid division by zero
        return num / den

    def _score_user_based(
        self, u: int, i: int, similar_users_indices: NDArray
    ) -> float:
        """
        Compute the recommendation score for given user u and item i
        """
        # Average rating for user u
        ru_mean = self.ratings[u, :].mean()
        res = 0
        sim_total = 0

        # Apply the scoring formula
        for v in similar_users_indices:
            # Similarity between user u and user v
            sim = self.similarity_matrix[u, v]

            # Average rating for user v
            rv_mean = self.ratings[v, :].mean()
            res += (self.ratings[v, i] - rv_mean) * sim
            sim_total += sim

        # Avoid division by zero
        if sim_total == 0:
            return 0.0
        return ru_mean + (res / sim_total)

    def _top_n_user_based(self, user_index: int, n: int, k=100) -> DataFrame:
        n_items = self.ratings.shape[1]

        # Retrieve the user's ratings
        user_ratings = self.ratings[user_index, :]

        # Find the missing ratings for the user
        unrated_items_indices = np.nonzero(user_ratings == 0)[0]

        # Find similar users
        similar_users_indices = np.argsort(self.similarity_matrix[user_index, :])[:k]

        # Compute the scores for all unrated items
        scores = np.zeros((n_items))
        for unrated_index in unrated_items_indices:
            scores[unrated_index] = self._score_user_based(
                user_index, unrated_index, similar_users_indices
            )

        top_n = np.argsort(scores)[::-1][:n].tolist()

        return self.data.get_movie_from_index(top_n)

    def _score_item_based(self, u: int, i: int, k: int) -> float:
        """
        Compute the score for given user u and item i
        """

        # Retrieve similar items
        similarities = self.similarity_matrix[:, i]
        similar_items = np.nonzero(similarities > 0)[0][:k]

        # Return zero if no similar items are found
        if not similar_items.size:
            return 0.0

        # Apply the scoring formula
        num = np.dot(
            similarities[similar_items],
            self.ratings[u, similar_items]
            - self.data.average_item_rating[similar_items],
        )
        den = np.sum(self.ratings[u, similar_items])

        # Avoid division by zero
        if den == 0:
            return 0.0
        return (num / den) + self.data.average_item_rating[i]

    def _top_n_item_based(self, user_index: int, n: int, k=100) -> DataFrame:
        n_items = self.ratings.shape[1]

        # Retrieve the user's ratings
        user_ratings = self.ratings[user_index, :]

        # Find the missing ratings for the user
        unrated_items_indices = np.nonzero(user_ratings == 0)[0]

        # Compute the scores for all unrated items
        scores = np.zeros((n_items))
        for unrated_index in unrated_items_indices:
            scores[unrated_index] = self._score_item_based(user_index, unrated_index, k)

        rec_indices = np.argsort(scores)[::-1][:n]
        return self.data.get_movie_from_index(rec_indices.tolist())

    def top_n_recommendations(self, user_id: int, n: int) -> DataFrame:
        """
        Given a user compute n movie recommendations based on the similarity matrix
        """

        user_index = self.data.id_to_index(user_id, "user")

        if self.kind == "item":
            return self._top_n_item_based(user_index, n)
        else:
            return self._top_n_user_based(user_index, n)
