from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.typing import NDArray


class Content_Based:
    """
    Given the items features, construct a ball tree through sklearn using Jaccard Distance
    as metric. Then queries the tree for the k most similar items given an item id.
    """

    def fit(self, items_df: DataFrame):
        self.items_df = items_df
        item_features = self._extract_features(self.items_df)
        self.model = NearestNeighbors(algorithm="ball_tree", metric="jaccard")
        self.model.fit(item_features)

    def get_recommendations(self, movie_id: int, k=5):
        """
        Given a movie id and optionally a number k of neighbors, return the neighbors
        of the selected movie
        """
        movie = self.items_df[self.items_df["movie_id"] == movie_id]
        features = self._extract_features(movie)
        distances, indices = self.model.kneighbors(features, n_neighbors=k + 1)
        indices = indices + 1

        mask = indices.flatten() != movie_id
        indices = indices.flatten()[mask]
        distances = distances.flatten()[mask]
        return self.retrieve_movies(indices)

    def _extract_features(self, row_df: DataFrame):
        """
        Compute the item features by slicing off the unneeded information
        """
        return row_df.iloc[:, 3:].values

    def retrieve_movies(self, movie_ids: int | NDArray[np.int64]) -> DataFrame:
        """
        Given a movie id or an array of ids, retrieve the full items information from
        the training dataframe, sorted by the order of the ids

        """
        indices = [movie_ids] if isinstance(movie_ids, int) else movie_ids.flatten()
        result_df = self.items_df[self.items_df["movie_id"].isin(indices)]
        result_df = result_df.set_index("movie_id")
        result_df = result_df.reindex(indices)
        result_df = result_df.reset_index()
        return result_df
