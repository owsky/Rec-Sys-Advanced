from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.typing import NDArray
from data import Data
from ..non_personalized import Non_Personalized


class Content_Based:
    """
    Given the items features, construct a ball tree through sklearn using Jaccard Distance
    as metric. Then queries the tree for the k most similar items given an item id.
    """

    def fit(self, data: Data):
        self.data = data
        item_features = self._extract_features(self.data.items)
        self.model = NearestNeighbors(algorithm="ball_tree", metric="jaccard")
        self.model.fit(item_features)
        self.np = Non_Personalized()
        self.np.fit(data)

    def get_n_movies_similar_to(self, movie_id: int, n=10):
        """
        Given a movie id and optionally a number k of neighbors, return the neighbors
        of the selected movie
        """
        movie = self.data.items[self.data.items["movie_id"] == movie_id]
        features = self._extract_features(movie)
        distances, indices = self.model.kneighbors(features, n_neighbors=n + 1)
        indices = indices + 1

        mask = indices.flatten() != movie_id
        indices = indices.flatten()[mask]
        distances = distances.flatten()[mask]
        return self.data.get_movies_from_ids(indices)

    def _extract_features(self, row_df: DataFrame) -> NDArray[np.int64]:
        """
        Compute the item features by slicing off the unneeded information
        """
        return row_df.iloc[:, 3:].values

    def get_top_n_recommendations(self, user_id: int, n=10) -> DataFrame:
        """
        Given a user id and, optionally, a number n of recommendations, retrieve n recommendations
        for the user
        """

        # Extract the ratings provided by the user
        user_ratings = self.data.ratings.data[
            self.data.ratings.row == self.data.id_to_index(user_id, "user")
        ]
        average_rating = user_ratings.mean()
        s = self._extract_features(self.data.items).shape[1]
        user_profile = np.zeros((s), dtype=bool)

        user_interactions = []
        how_many = 5
        for index, rating in enumerate(sorted(user_ratings, reverse=True)):
            if rating == 0:
                break
            movie_id = self.data.index_to_id(index, "movie")
            user_interactions.append(movie_id)
            if rating >= average_rating and how_many > 0:
                movie_genres = self._extract_features(
                    self.data.get_movies_from_ids(movie_id)
                )
                user_profile = np.bitwise_or(user_profile, movie_genres.astype(bool))
                how_many -= 1

        if np.sum(user_profile) == 0:
            return self.np.get_n_most_popular(user_id, n)

        # Find similar items based on the user profile
        distances, indices = self.model.kneighbors(
            user_profile, n_neighbors=n + len(user_interactions)
        )

        # Remove from indices the ids found in user_interactions
        indices = np.array(
            [id for id in indices.flatten() if id not in user_interactions]
        )[:n]

        return self.data.get_movies_from_ids(indices)
