import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import ndcg_score
from data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, spmatrix
from utils import lists_str_join
from models import Non_Personalized


def z_score_norm(ratings):
    # Calculate mean and standard deviation of non-zero ratings
    non_zero_ratings = ratings[ratings != 0]
    mean_rating = np.mean(non_zero_ratings)
    std_rating = np.std(non_zero_ratings)

    # Apply z-score normalization only to non-zero values
    normalized_ratings = np.zeros_like(ratings, dtype=np.float64)
    non_zero_indices = ratings != 0
    normalized_ratings[non_zero_indices] = (
        ratings[non_zero_indices] - mean_rating
    ) / std_rating

    return normalized_ratings


class Content_Based:
    def __init__(self, data: Data):
        self.data = data
        self.train: NDArray[np.float64] = z_score_norm(self.data.train.toarray())
        self.movies = self.data.movies
        self.np = Non_Personalized()
        self.np.fit(data)

    def fit(self):
        """
        Fit the Tfid and NearestNeighbors models, then create the user profiles
        """

        self.vec_model = TfidfVectorizer(
            vocabulary=list(
                set(
                    self.data.genre_vocabulary.tolist()
                    + self.data.tags_vocabulary.tolist()
                )
            ),
            sublinear_tf=True,
            smooth_idf=True,
        )

        # Build the train dataset for Nearest Neighbors
        train = []
        current = 0
        self.sim_index_to_movie_id = {}
        for _, (movieId, _, genres, tags) in self.data.movies.iterrows():
            train.append(lists_str_join(genres, tags))
            self.sim_index_to_movie_id[current] = movieId
            current += 1

        # Fit transform the Tfidf model and use its ouput to train the Nearest Neighbors model
        m = self.vec_model.fit_transform(train)
        self.knn_model = NearestNeighbors(metric="cosine").fit(m)

        print("Creating the user profiles")
        n_users = self.train.shape[0]
        self.cold_users = []
        self.user_profiles = []
        for user_index in range(n_users):
            profile = self._create_user_profile(user_index)
            if isinstance(profile, int):
                self.cold_users.append(profile)
                self.user_profiles.append(None)
            else:
                self.user_profiles.append(profile)

    def _get_movie_vector(self, movie_index: int) -> spmatrix:
        """
        Given a movie index compute the respective tfidf matrix
        """
        movie_id = self.data.index_to_id(movie_index, "item")
        movie = self.data.get_movies_from_ids([movie_id])

        movie_genres = movie["genres"].values
        movie_tags = movie["tags"].values

        movie_genres_tags = lists_str_join(movie_genres.tolist(), movie_tags.tolist())
        return self.vec_model.transform([movie_genres_tags])

    def _create_user_profile(self, user_index: int, k=4) -> int | csr_matrix:
        """
        Create user profile by averaging the k most liked movies' genres and tags
        If a user has no liked movies return the index and let the caller deal with it
        """
        ratings = self.train[user_index, :]
        rated_indices = np.nonzero(ratings)[0]

        # Cold user
        if len(rated_indices) < 30:
            return user_index

        user_bias = self.data.average_user_rating[user_index]
        liked_indices = np.nonzero(ratings - user_bias > 0)[0]
        if len(liked_indices) == 0:
            liked_indices = rated_indices
        # k = int(0.01 * len(liked_indices)) + 1
        # Sort rated items by user's ratings in descending order
        sorted_indices = sorted(
            liked_indices, key=lambda index: ratings[index], reverse=True  # type: ignore
        )[:k]

        # Collect movie vectors and corresponding weights based on ratings
        movie_vectors = []
        weights = []

        for index in sorted_indices:
            movie_vector = np.array(self._get_movie_vector(index))
            weight = ratings[index]
            movie_vectors.append(movie_vector)
            weights.append(weight)

        # Convert lists to arrays
        movie_vectors = np.stack(movie_vectors)
        weights = np.array(weights)

        # Apply weighted averaging
        weighted_average = np.average(movie_vectors, axis=0, weights=weights)

        # Convert back to csr_matrix
        user_profile = csr_matrix(weighted_average)

        return user_profile

    def get_top_n_recommendations(self, user_index: int, n=10) -> list[int]:
        """
        Compute the top n recommendations for given user index
        """
        # print(user_index)
        if user_index in self.cold_users:
            user_id = self.data.user_index_to_id[user_index]
            return self.np.get_n_highest_rated(user_id, n).tolist()

        user_profile = self.user_profiles[user_index]
        already_watched_indices = self.train[user_index, :].nonzero()[0]

        neighbors = self.knn_model.kneighbors(
            user_profile, n + len(already_watched_indices), False
        )
        movie_ids = [
            self.sim_index_to_movie_id[neighbor]
            for neighbor in neighbors[0]
            if neighbor not in already_watched_indices
        ]

        return movie_ids[:n]

    def _average_reciprocal_hit_rank(
        self, recommended_indices: list[int], relevant_items_indices: list[int]
    ):
        """
        Compute the average reciprocal hit rank
        """
        hits = np.intersect1d(relevant_items_indices, recommended_indices)
        ranks = [np.where(recommended_indices == hit)[0][0] + 1 for hit in hits]
        if len(ranks) == 0:
            return 0
        return np.mean([1 / rank for rank in ranks])

    def _ndcg(self, recommended_indices: list[int], relevant_items_indices: list[int]):
        """
        Compute the normalized discounted cumulative gain
        """
        binary_relevance = [
            int(idx in relevant_items_indices) for idx in recommended_indices
        ]
        ideal_relevance = sorted(binary_relevance, reverse=True)
        return ndcg_score(np.array([ideal_relevance]), np.array([binary_relevance]))

    def accuracy_metrics(self):
        """
        Compute all accuracy metrics using the test set
        """
        n_users = self.data.test.shape[0]
        test = self.data.test.toarray()

        precisions = []
        recalls = []
        f1_scores = []
        arhrs = []
        ndcgs = []
        for user_index in range(n_users):
            ratings = test[user_index, :]
            if ratings.any():
                recommended_ids = self.get_top_n_recommendations(user_index, 10)
                recommended_indices = [
                    self.data.item_id_to_index[id] for id in recommended_ids
                ]

                relevant_items_indices = np.nonzero(ratings)[0].tolist()

                n_relevant = len(relevant_items_indices)

                if n_relevant == 0:
                    n_relevant = 1

                relevant_recommended = np.intersect1d(
                    relevant_items_indices, recommended_indices
                )

                precision = len(relevant_recommended) / len(recommended_indices)
                recall = len(relevant_recommended) / n_relevant
                arhr = self._average_reciprocal_hit_rank(
                    recommended_indices, relevant_items_indices
                )
                ndcg = self._ndcg(recommended_indices, relevant_items_indices)
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                arhrs.append(arhr)
                ndcgs.append(ndcg)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1_scores = np.array(f1_scores)
        arhrs = np.array(arhrs)
        ndcgs = np.array(ndcgs)

        return (
            np.mean(precisions),
            np.mean(recalls),
            np.mean(f1_scores),
            np.mean(arhrs),
            np.mean(ndcgs),
        )
