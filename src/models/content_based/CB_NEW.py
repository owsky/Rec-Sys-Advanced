import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import ndcg_score
from data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, spmatrix
from utils import lists_str_join


class Content_Based:
    def __init__(self, data: Data):
        self.data = data
        self.train: NDArray[np.float64] = self.data.train.toarray()
        self.movies = self.data.movies

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
        user_profiles_tmp: list[int | csr_matrix] = []
        for user_index in range(n_users):
            user_profiles_tmp.append(self._create_user_profile(user_index))
        average_user_profile: NDArray[np.float64] = np.stack(
            [x.toarray() for x in user_profiles_tmp if not isinstance(x, int)]
        ).mean(axis=0)
        self.user_profiles: list[csr_matrix] = [
            csr_matrix(average_user_profile) if isinstance(x, int) else x
            for x in user_profiles_tmp
        ]

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
        ratings = self.train[user_index, :].tolist()
        rated_indices = np.nonzero(ratings)[0]

        if len(rated_indices) == 0:
            # No liked movies
            return user_index

        # Sort rated items by user's ratings in descending order
        sorted_indices = sorted(
            rated_indices, key=lambda index: ratings[index], reverse=True
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
        user_profile = self.user_profiles[user_index]
        already_watched_indices = self.train[user_index, :].nonzero()[0]

        distances, neighbors = self.knn_model.kneighbors(
            user_profile, n + len(already_watched_indices), True
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

        def aux(user_index: int):
            n = 10
            recommended_ids = self.get_top_n_recommendations(user_index, n)
            recommended_indices = [
                self.data.item_id_to_index[id] for id in recommended_ids
            ]

            ratings = test[user_index, :]
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

            return precision, recall, arhr, ndcg

        precisions = []
        recalls = []
        arhrs = []
        ndcgs = []
        for user_index in range(n_users):
            if test[user_index, :].nonzero()[0].size > 0:
                res = aux(user_index)
                if res:
                    precision, recall, arhr, ndcg = res
                    precisions.append(precision)
                    recalls.append(recall)
                    arhrs.append(arhr)
                    ndcgs.append(ndcg)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        arhrs = np.array(arhrs)
        ndcgs = np.array(ndcgs)

        return (np.mean(precisions), np.mean(recalls), np.mean(arhrs), np.mean(ndcgs))
