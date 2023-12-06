from data import Data
from models import Hybrid


def hb(data: Data):
    model = Hybrid().fit(data)
    model.pretty_print_accuracy_top_n()

    # user_id = 1
    # print(f"\nGetting recommendations for user {user_id}")

    # n = 10
    # likes = data.get_liked_movies_indices(user_id, "train")[:n]

    # print(f"User {user_id} previously liked:")
    # print(data.get_movie_from_indices(likes[:n]))

    # user_index = data.user_id_to_index[user_id]
    # recs = model.top_n(user_index, n)
    # print(f"Recommendations for user {user_id}")
    # print(data.get_movies_from_ids(recs))
