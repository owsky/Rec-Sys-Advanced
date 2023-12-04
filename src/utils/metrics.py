import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from sklearn.metrics import ndcg_score


def precision_at_k(
    relevant_items: list[int] | NDArray[np.int64],
    recommended_items: list[int] | NDArray[np.int64],
    k: int,
):
    relevant_recommended = np.intersect1d(relevant_items[:k], recommended_items[:k])
    return len(relevant_recommended) / k


def recall_at_k(
    relevant_items: list[int] | NDArray[np.int64],
    recommended_items: list[int] | NDArray[np.int64],
    k: int,
):
    relevant_recommended = np.intersect1d(relevant_items[:k], recommended_items[:k])
    return len(relevant_recommended) / len(relevant_items)


def average_reciprocal_hit_rank(
    relevant_items: list[int] | NDArray[np.int64],
    recommended_items: list[int] | NDArray[np.int64],
):
    hits = np.intersect1d(relevant_items, recommended_items)
    ranks = [np.where(recommended_items == hit)[0][0] + 1 for hit in hits]
    if len(ranks) == 0:
        return 0
    return np.mean([1 / rank for rank in ranks])


def normalized_discounted_cumulative_gain(
    relevant_items: list[int] | NDArray[np.int64],
    recommended_items: list[int] | NDArray[np.int64],
):
    """
    Compute the normalized discounted cumulative gain
    """
    binary_relevance = [int(idx in relevant_items) for idx in recommended_items]
    ideal_relevance = sorted(binary_relevance, reverse=True)
    return ndcg_score(np.array([ideal_relevance]), np.array([binary_relevance]))


def get_most_liked_indices(
    user_ratings: NDArray[np.float64], user_bias: float, k: int | None = None
) -> list[int]:
    if len(user_ratings[user_ratings != 0]) == 0:
        return []
    liked_indices = np.flatnonzero(user_ratings - user_bias > 0)
    if len(liked_indices) == 0:
        liked_indices = np.flatnonzero(user_ratings)
    sorted_indices = sorted(liked_indices, key=lambda x: user_ratings[x], reverse=True)  # type: ignore
    return sorted_indices if k is None else sorted_indices[:k]


# def get_relevant(
#     user_ratings: NDArray[np.float64] | csr_array, avg_rating: float
# ) -> NDArray[np.int64]:
#     if isinstance(user_ratings, csr_array):
#         ratings = user_ratings.toarray()[0]
#     else:
#         ratings = user_ratings

#     return np.array(get_most_liked_indices(ratings, avg_rating))
