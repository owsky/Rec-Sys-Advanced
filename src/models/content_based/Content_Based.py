from typing_extensions import Self
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import ndcg_score
from data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix, coo_array, csr_array
from utils import lists_str_join
from utils import recall_at_k, precision_at_k
from joblib import Parallel, delayed
from tqdm import tqdm
from models.non_personalized import Highest_Rated


def z_score_norm(ratings: coo_array):
    # Initialize an empty COO matrix for normalized values
    normalized_data = []
    normalized_rows = []
    normalized_cols = []

    # Iterate through each row
    for i in range(ratings.shape[0]):
        # Extract non-zero elements in the row
        row_elements = ratings.data[ratings.row == i]
        if row_elements.any():
            # Compute mean and standard deviation excluding zero values
            mean_value = np.mean(row_elements)
            std_value = np.std(row_elements)

            # Normalize non-zero elements using z-score
            if std_value == 0:
                std_value = 1
            normalized_values = (row_elements - mean_value) / std_value

            # Append normalized values to the result COO matrix
            normalized_data.extend(normalized_values)
            normalized_rows.extend([i] * len(normalized_values))
            normalized_cols.extend(ratings.col[ratings.row == i])

    normalized_coo_matrix = coo_array(
        (normalized_data, (normalized_rows, normalized_cols)), shape=ratings.shape
    )

    return normalized_coo_matrix


class Content_Based:
    def __init__(self, data: Data):
        self.data = data
        # self.train = z_score_norm(data.train).tocsr()
        self.train = data.train.tocsr()
        self.movies = self.data.movies
        self.np = Highest_Rated().fit(data)

    def fit(self) -> Self:
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

        results = [
            result
            for result in Parallel(n_jobs=-1, backend="loky")(
                delayed(self._create_user_profile)(user_index)
                for user_index in tqdm(range(n_users))
            )
            if result is not None
        ]

        for profile in results:
            if isinstance(profile, int):
                self.cold_users.append(profile)
                self.user_profiles.append(None)
            else:
                self.user_profiles.append(profile)

        return self

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

    def _create_user_profile(self, user_index: int) -> int | NDArray:
        """
        Create user profile by averaging the k most liked movies' genres and tags
        If a user has no liked movies return the index and let the caller deal with it
        """
        user_id = self.data.user_index_to_id[user_index]
        user_ratings = self.data.get_user_ratings(user_id, "train")

        k = int(0.4 * len(user_ratings)) + 1
        user_likes = self.data.get_liked_movies_indices(user_id, "train")[:k]

        # Collect movie vectors and corresponding weights based on ratings
        movie_vectors = np.array(
            [self._get_movie_vector(index) for index in user_likes]
        )
        weights = user_ratings[user_likes]

        try:
            # Apply weighted averaging
            weighted_average = np.average(movie_vectors, axis=0, weights=weights)
        except ZeroDivisionError:
            return user_index

        return weighted_average

    def get_top_n_recommendations(self, user_index: int, n=10) -> list[int]:
        """
        Compute the top n recommendations for given user index
        """
        if user_index in self.cold_users:
            return self.np.top_n(user_index, n).tolist()

        user_profile = self.user_profiles[user_index]
        already_watched_indices = csr_array(self.data.train.getrow(user_index)).indices

        n_movies = self.data.train.shape[1]
        max_neighbors = min(n + len(already_watched_indices), n_movies)

        neighbors = self.knn_model.kneighbors(user_profile, max_neighbors, False)

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

    def accuracy_metrics(self, n=10):
        n_users = self.train.shape[0]

        precisions = []
        recalls = []
        f1_scores = []
        arhrs = []
        ndcgs = []
        for user_index in range(n_users):
            user_ratings = self.data.test.getrow(user_index).toarray()[0]  # type: ignore
            user_bias = self.data.average_user_rating[user_index]
            # relevant = get_most_liked_indices(user_ratings, user_bias)
            user_id = self.data.user_index_to_id[user_index]
            relevant = self.data.get_liked_movies_indices(user_id, "test")
            recommended = [
                self.data.item_id_to_index[x]
                for x in self.get_top_n_recommendations(user_index, n)
            ]

            if len(relevant) >= 1 and len(recommended) >= 1:
                precision = precision_at_k(relevant, recommended, n)
                recall = recall_at_k(relevant, recommended, n)
                arhr = self._average_reciprocal_hit_rank(recommended, relevant)
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
                ndcg = self._ndcg(recommended, relevant)
                precisions.append(precision)
                recalls.append(recall)
                arhrs.append(arhr)
                f1_scores.append(f1)
                ndcgs.append(ndcg)

        return (
            np.mean(precisions),
            np.mean(recalls),
            np.mean(f1_scores),
            np.mean(arhrs),
            np.mean(ndcgs),
        )
