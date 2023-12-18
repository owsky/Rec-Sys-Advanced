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


def reciprocal_rank(
    relevant_items: list[int] | NDArray[np.int64],
    recommended_items: list[int] | NDArray[np.int64],
):
    for i, item in enumerate(recommended_items, 1):
        if item in relevant_items:
            return 1 / i
    return 0


def normalized_discounted_cumulative_gain(
    relevant_items: list[int] | NDArray[np.int64],
    recommended_items: list[int] | NDArray[np.int64],
):
    y_true = np.array(
        [[1 if item in relevant_items else 0 for item in recommended_items]]
    )
    y_score = np.array([recommended_items])
    return ndcg_score(y_true, y_score)
