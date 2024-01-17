from data import Data
from models import Hybrid, Word2VecSession


def hb(data: Data):
    hb_model = Hybrid(data).fit(False, False, 0.77)
    hb_model.pretty_print_accuracy_top_n()

    user_id = 1
    print(f"\nGetting recommendations for user {user_id}")

    n = 15
    likes = data.get_liked_movies_indices(user_id, True, "train")[:n]

    print(f"User {user_id} previously liked:")
    print(data.get_movie_from_indices(likes[:n]))

    user_index = data.user_id_to_index[user_id]
    recs = hb_model.top_n(user_index, n)
    print(f"Recommendations for user {user_id}")
    print(data.get_movies_from_ids(recs))

    w2v_model = Word2VecSession(data).fit(biased=True, window=5, vector_size=5)
    w2v_model.pretty_print_accuracy_top_n()
    recs = w2v_model.top_n(user_index, n)
    print(f"Recommendations for user {user_id}")
    print(data.get_movies_from_ids(recs))
