import numpy as np
from numpy.typing import NDArray
from data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix, csr_array
from ..Recommender_System import Recommender_System
from utils import lists_str_join
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from models.non_personalized import Most_Popular


class Content_Based(Recommender_System):
    def __init__(self, data: Data):
        super().__init__(data, "Content Based")

    def fit(
        self, by_timestamp: bool, biased: bool, like_perc: float, silent=False, cv=False
    ):
        """
        Fit the Tfid and NearestNeighbors models, then create the user profiles
        """
        self.is_fit = True
        self.is_biased = biased
        self.movies = self.data.movies
        self.np = Most_Popular(self.data).fit(cv)
        self.train_set = (
            self.data.interactions_cv_train if cv else self.data.interactions_train
        )

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
        m = self.vec_model.fit_transform(train)
        self.knn_model = NearestNeighbors(metric="cosine").fit(m)

        n_users = self.train_set.shape[0]
        self.cold_users = []
        self.user_profiles = []

        results = [
            result
            for result in Parallel(n_jobs=-1, backend="sequential")(
                delayed(self._create_user_profile)(
                    user_index, by_timestamp, biased, like_perc
                )
                for user_index in tqdm(
                    range(n_users),
                    leave=False,
                    desc="Fitting the Content Based model...",
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

    def _get_movie_vector(self, movie_index: int) -> spmatrix:
        """
        Given a movie index compute the respective tfidf matrix
        """
        movie_id = self.data.item_index_to_id[movie_index]
        movie = self.data.get_movies_from_ids([movie_id])

        movie_genres = movie["genres"].values
        movie_tags = movie["tags"].values

        movie_genres_tags = lists_str_join(movie_genres.tolist(), movie_tags.tolist())
        return self.vec_model.transform([movie_genres_tags])

    def _create_user_profile(
        self, user_index: int, by_timestamp: bool, biased: bool, like_perc: float
    ) -> int | NDArray:
        """
        Create user profile by averaging the k most liked movies' genres and tags
        If a user has no liked movies return the index and let the caller deal with it
        """
        user_id = self.data.user_index_to_id[user_index]
        k = int(like_perc * self.data.get_ratings_count(user_id)) + 1
        if by_timestamp:
            user_likes = self.data.get_weighed_liked_movie_indices(user_id, biased)
            if len(user_likes) == 0:
                return user_index
            user_likes = user_likes[:k]
            movie_vectors = np.array(
                [self._get_movie_vector(index) for (index, _, _) in user_likes]
            )
            weights = [rating * weight for (_, rating, weight) in user_likes]
        else:
            user_ratings = self.data.get_user_ratings(user_id, "train")
            user_likes = self.data.get_liked_movies_indices(user_id, biased, "train")
            if len(user_likes) == 0:
                return user_index
            user_likes = user_likes[:k]
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

    def top_n(self, user_index: int, n=10) -> list[int]:
        """
        Compute the top n recommendations for given user index
        """
        if user_index in self.cold_users:
            return self.np.top_n(user_index, n).tolist()

        user_profile = self.user_profiles[user_index]
        already_watched_indices = csr_array(self.train_set.getrow(user_index)).indices

        n_movies = self.train_set.shape[1]
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
