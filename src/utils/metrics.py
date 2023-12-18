import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import ndcg_score


def precision(
    relevant_items: list[int] | NDArray[np.int64],
    recommended_items: list[int] | NDArray[np.int64],
):
    rel_set = set(relevant_items)
    rec_set = set(recommended_items)
    tp = len(rel_set.intersection(rec_set))
    fp = len(rec_set.difference(rel_set))
    return tp / (fp + tp) if fp + tp != 0 else 0


def recall(
    relevant_items: list[int] | NDArray[np.int64],
    recommended_items: list[int] | NDArray[np.int64],
):
    rel_set = set(relevant_items)
    rec_set = set(recommended_items)
    tp = len(rel_set.intersection(rec_set))
    fn = len(rel_set.difference(rec_set))
    return tp / (fn + tp) if fn + tp != 0 else 0


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


def f1_score(precision: float, recall: float):
    return (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )


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
    binary_relevance = [int(idx in relevant_items) for idx in recommended_items]
    ideal_relevance = sorted(binary_relevance, reverse=True)
    return ndcg_score(np.array([ideal_relevance]), np.array([binary_relevance]))
