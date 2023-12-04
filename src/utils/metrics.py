import numpy as np
from numpy.typing import NDArray
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
