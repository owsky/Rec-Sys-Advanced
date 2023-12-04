from data import Data
from models import Content_Based


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
    print(f"\nGetting recommendations for user {user_id}")

    n = 10
    likes = data.get_liked_movies_indices(user_id, "train")[:n]

    print(f"User {user_id} previously liked:")
    print(data.get_movie_from_indices(likes[:n]))

    user_index = data.user_id_to_index[user_id]
    recs = model.get_top_n_recommendations(user_index, n)
    print(f"Recommendations for user {user_id}")
    print(data.get_movies_from_ids(recs))

    pretty_print_accuracy_top_n(model, 30)
