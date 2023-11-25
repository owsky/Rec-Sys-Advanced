from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from data import Data
from ..non_personalized import Non_Personalized
from joblib import Parallel, delayed
from sklearn.metrics import ndcg_score


class Content_Based:
    """
    Given the items features, construct a ball tree through sklearn using Jaccard Distance
    as metric. Then queries the tree for the k most similar items given an item id.
    """

    def fit(self, data: Data):
        self.data = data
        self.train = data.train.todense()
        item_features = self._extract_features(self.data.items)
        self.model = NearestNeighbors(algorithm="ball_tree", metric="jaccard")
        self.model.fit(item_features)
        self.np = Non_Personalized()
        self.np.fit(data)

        n_users = self.train.shape[0]
        self.user_profiles = []
        for user_index in range(n_users):
            self.user_profiles.append(self._create_user_profile(user_index))

    def _extract_features(self, row_df: DataFrame) -> NDArray[np.int64]:
        """
        Compute the item features by slicing off the unneeded information
        """
        return row_df.iloc[:, 3:].values

    def _create_user_profile(self, user_index: int, k=10):
        user_bias = self.data.average_user_rating[user_index]
        ratings = self.train[user_index, :]
        filtered_ratings = np.nonzero(ratings - user_bias > 0)[0]
        most_liked_indices = sorted(
            filtered_ratings, key=lambda index: ratings[index], reverse=True  # type: ignore
        )[:k]

        movies = self.data.get_movie_from_indices(most_liked_indices)
        genres_array = movies.iloc[:, 3:].values
        return genres_array

    def get_top_n_recommendations(
        self, user_index: int, n=10, ret_df=True
    ) -> DataFrame | list[int]:
        user_profile = self.user_profiles[user_index]

        all_neighbors = []
        all_distances = []
        for genres in user_profile:
            distances, neighbors = self.model.kneighbors(
                genres.reshape(1, -1), 10, True
            )
            all_neighbors.append(neighbors.flatten().tolist())
            all_distances.append(distances.flatten().tolist())

        already_rated_indices = np.nonzero(self.train[user_index, :])[0]

        all_neighbors = np.array(all_neighbors).flatten()
        all_distances = np.array(all_distances).flatten()

        recommendations = []
        for neighbor, _ in sorted(
            zip(all_neighbors, all_distances), key=lambda x: x[1]
        ):
            if neighbor not in already_rated_indices:
                recommendations.append(neighbor)

        if ret_df:
            return self.data.get_movie_from_indices(recommendations[:n])
        else:
            return recommendations[:n]

    def _hit_rate(
        self, recommended_indices: list[int], relevant_items_indices: list[int]
    ):
        hits = np.intersect1d(relevant_items_indices, recommended_indices)
        return len(hits) / len(recommended_indices)

    def _average_reciprocal_hit_rank(
        self, recommended_indices: list[int], relevant_items_indices: list[int]
    ):
        hits = np.intersect1d(relevant_items_indices, recommended_indices)
        ranks = [np.where(recommended_indices == hit)[0][0] + 1 for hit in hits]
        if len(ranks) == 0:
            return 0
        return np.mean([1 / rank for rank in ranks])

    def _ndcg(self, recommended_indices: list[int], relevant_items_indices: list[int]):
        binary_relevance = [
            int(idx in relevant_items_indices) for idx in recommended_indices
        ]
        ideal_relevance = sorted(binary_relevance, reverse=True)
        return ndcg_score(np.array([ideal_relevance]), np.array([binary_relevance]))

    def accuracy_metrics(self):
        n_users = self.data.test.shape[0]
        test = self.data.test.todense()

        def aux(user_index: int):
            recommended_indices = self.get_top_n_recommendations(
                user_index, 20, ret_df=False
            )
            if isinstance(recommended_indices, list):
                user_bias = self.data.average_user_rating[user_index]
                relevant_items_indices = np.nonzero(
                    test[user_index, :] - user_bias > 0
                )[0].tolist()

                if len(recommended_indices) < 20 or len(relevant_items_indices) < 50:
                    return None

                precision = len(
                    np.intersect1d(relevant_items_indices, recommended_indices)
                ) / len(recommended_indices)
                recall = len(
                    np.intersect1d(relevant_items_indices, recommended_indices)
                ) / len(relevant_items_indices)
                hit_rate = self._hit_rate(recommended_indices, relevant_items_indices)
                arhr = self._average_reciprocal_hit_rank(
                    recommended_indices, relevant_items_indices
                )

                ndcg = self._ndcg(recommended_indices, relevant_items_indices)

                return precision, recall, hit_rate, arhr, ndcg

        results = [
            result
            for result in Parallel(n_jobs=-1, backend="sequential")(
                delayed(aux)(user_index)
                for user_index in tqdm(
                    range(n_users), desc="Computing accuracy metrics"
                )
            )
            if result is not None
        ]
        precision, recall, hit_rate, arhr, ndcg = zip(*results)

        return (
            np.mean(precision),
            np.mean(recall),
            np.mean(hit_rate),
            np.mean(arhr),
            np.mean(ndcg),
        )
