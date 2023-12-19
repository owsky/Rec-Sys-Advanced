import numpy as np
from numpy.typing import NDArray
from data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_array, csr_matrix
from ..Recommender_System import Recommender_System
from utils import lists_str_join
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from models.non_personalized import Most_Popular


class Hybrid(Recommender_System):
    def __init__(self, data: Data):
        super().__init__(data, "Hybrid")

    def _extract_item_features(self) -> NDArray[np.float64]:
        self.vec_model = TfidfVectorizer(
            vocabulary=list(
                set(
                    self.data.genre_vocabulary.tolist()
                    + self.data.tags_vocabulary.tolist()
                )
            ),
            sublinear_tf=True,
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
        return csr_matrix(self.vec_model.fit_transform(train)).toarray()

    def _combine_features(
        self, items_features: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        i_features = items_features
        n_users, n_movies = self.data.interactions_train.shape
        combined_features = np.zeros((n_movies, len(i_features[0]) + n_users))

        for sim_index in range(i_features.shape[0]):
            movie_index = self.data.item_id_to_index[
                self.sim_index_to_movie_id[sim_index]
            ]
            movie_ratings = self.data.interactions_train_numpy[:, movie_index]
            movie_features = i_features[sim_index]
            comb = movie_features.tolist() + movie_ratings.tolist()
            combined_features[movie_index] = comb

        return np.array(combined_features)

    def fit(self, by_timestamp: bool, biased: bool, like_perc: float, silent=False):
        """
        Fit the Tfid and NearestNeighbors models, then create the user profiles
        """
        self.is_fit = True
        self.is_biased = biased
        self.np = Most_Popular(self.data).fit(silent)

        item_features = self._extract_item_features()
        self.movie_vectors = self._combine_features(item_features)
        self.knn_model = NearestNeighbors(metric="cosine").fit(self.movie_vectors)

        n_users = self.data.interactions_train.shape[0]
        self.cold_users = []
        self.user_profiles = []

        results = [
            result
            for result in Parallel(n_jobs=-1, backend="loky")(
                delayed(self._create_user_profile)(
                    user_index, by_timestamp, biased, like_perc
                )
                for user_index in tqdm(
                    range(n_users),
                    desc="Fitting the Hybrid model...",
                    leave=False,
                    disable=silent,
                    dynamic_ncols=True,
                )
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

    def _create_user_profile(
        self, user_index: int, by_timestamp: bool, biased: bool, like_perc: float
    ) -> int | NDArray[np.float64]:
        """
        Create user profile by averaging the k most liked movies' genres and tags
        If a user has no liked movies return the index and let the caller deal with it
        """
        user_id = self.data.user_index_to_id[user_index]
        if by_timestamp:
            user_ratings = self.data.get_weighed_user_ratings(user_id)

            k = int(like_perc * len(user_ratings)) + 1
            user_likes = self.data.get_weighed_liked_movie_indices(user_id, biased)[:k]
            movie_vectors = np.array(
                [self.movie_vectors[index] for (index, _, _) in user_likes]
            )
            weights = [rating * weight for (_, rating, weight) in user_likes]
        else:
            user_ratings = self.data.get_user_ratings(user_id, "train")

            k = int(like_perc * np.count_nonzero(user_ratings)) + 1
            user_likes = self.data.get_liked_movies_indices(user_id, biased, "train")[
                :k
            ]

            # Collect movie vectors and corresponding weights based on ratings
            movie_vectors = np.array(
                [self.movie_vectors[index] for index in user_likes]
            )
            weights = user_ratings[user_likes]

        try:
            weighted_average = np.average(movie_vectors, axis=0, weights=weights)
        except ZeroDivisionError:
            return user_index

        return weighted_average

    def top_n(self, user_index: int, n=10) -> list[int]:
        """
        Compute the top n recommendations for given user index
        """
        if user_index in self.cold_users:
            return self.np.top_n(user_index, n).tolist()

        user_profile = self.user_profiles[user_index].reshape(1, -1)
        already_watched_indices = csr_array(
            self.data.interactions_train.getrow(user_index)
        ).indices

        n_movies = self.data.interactions_train.shape[1]
        max_neighbors = min(n + len(already_watched_indices), n_movies)

        neighbors = self.knn_model.kneighbors(user_profile, max_neighbors, False)

        movie_ids = [
            self.sim_index_to_movie_id[neighbor]
            for neighbor in neighbors[0]
            if neighbor not in already_watched_indices
        ]

        return movie_ids[:n]

    def predict(self):
        raise RuntimeError(f"Model {self.__class__.__name__} cannot predict ratings")

    def _predict_all(self):
        raise RuntimeError(f"Model {self.__class__.__name__} cannot predict ratings")
