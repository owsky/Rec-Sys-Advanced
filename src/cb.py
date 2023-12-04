import numpy as np
from numpy.typing import NDArray
from data import Data
from models import Content_Based
from utils import get_most_liked_indices


def pretty_print_accuracy_top_n(m: Content_Based, n=30):
    precision, recall, f1, arhr, ndcg = m.accuracy_metrics(n)
    print(
        f"Precision@k: {precision:.3f}, Recall@k: {recall:.3f}, ",
        f"F1: {f1:.3f}, Average Reciprocal Hit Rank: {arhr:.3f}, ",
        f"Normalized Discounted Cumulative Gain: {ndcg:.3f}\n",
    )


def cb(data: Data):
    print("\nContent Based recommender:")
    model = Content_Based(data).fit()

    user_id = 1
    user_index = data.user_id_to_index[user_id]
    print(f"\nGetting recommendations for user {user_id}")

    user_ratings: NDArray[np.float64] = data.train.getrow(user_index).data  # type: ignore
    user_bias = data.average_user_rating[user_index]
    likes = get_most_liked_indices(user_ratings, user_bias)

    print(f"User {user_id} previously liked:")
    print(data.get_movie_from_indices(likes[:10]))

    recs = model.get_top_n_recommendations(user_index, 10)
    print(data.get_movies_from_ids(recs))

    pretty_print_accuracy_top_n(model, 30)
