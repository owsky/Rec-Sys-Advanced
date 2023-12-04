from typing_extensions import Self
import numpy as np
from numpy.typing import NDArray
from data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
from utils import lists_str_join
from models.non_personalized import Highest_Rated
from utils.metrics import get_most_liked_indices
from utils.z_score_norm import z_score_norm
from joblib import Parallel, delayed
from tqdm import tqdm
from ..Recommender_System import Recommender_System
from scipy.sparse import csr_array


class Content_Based(Recommender_System):
    ratings_train: csr_array

    def __init__(self, data: Data):
        super().__init__("Content Based")
        self.data = data
        self.ratings_train = z_score_norm(data.train).tocsr()
        self.movies = self.data.movies
        self.np = Highest_Rated()
        self.np.fit(data)

    def fit(self) -> Self:
        """
        Fit the Tfid and NearestNeighbors models, then create the user profiles
        """
        print("\nFitting the Content Based model...")

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
        n_users = self.ratings_train.shape[0]
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
        user_ratings: NDArray[np.float64] = self.ratings_train.getrow(user_index).data  # type: ignore
        user_bias: float = 0  # self.data.average_user_rating[user_index]
        k = int(0.4 * len(user_ratings)) + 1
        user_likes = get_most_liked_indices(user_ratings, user_bias, k)

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

    def top_n(self, user_index: int, n=10) -> list[int] | NDArray[np.int64]:
        """
        Compute the top n recommendations for given user index
        """
        if user_index in self.cold_users:
            return self.np.top_n(user_index, n)

        user_profile = self.user_profiles[user_index]
        already_watched_indices = self.ratings_train.getrow(user_index).nonzero()[0]

        neighbors = self.knn_model.kneighbors(
            user_profile, n + len(already_watched_indices), False
        )
        movie_ids = [
            self.sim_index_to_movie_id[neighbor]
            for neighbor in neighbors[0]
            if neighbor not in already_watched_indices
        ]

        return [self.data.item_id_to_index[x] for x in movie_ids[:n]]

    def predict(self):
        raise RuntimeError(f"Model {self.__class__.__name__} cannot predict ratings")

    def _predict_all(self):
        raise RuntimeError(f"Model {self.__class__.__name__} cannot predict ratings")

    def crossvalidation_hyperparameters(self):
        raise RuntimeError(
            f"Model {self.__class__.__name__} has no hyperparameters to crossvalidate"
        )
