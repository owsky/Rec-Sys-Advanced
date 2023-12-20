from data import Data
from models import Hybrid, Word2VecSession


def hb(data: Data):
    # Hybrid(data).fit(False, False, 0.4).pretty_print_accuracy_top_n()
    # Hybrid(data).fit(True, False, 0.4).pretty_print_accuracy_top_n()
    # Hybrid(data).fit(False, True, 0.4).pretty_print_accuracy_top_n()
    # Hybrid(data).fit(True, True, 0.4).pretty_print_accuracy_top_n()
    # Hybrid(data).fit(False, False, 1.0).pretty_print_accuracy_top_n()
    Hybrid(data).fit(False, False, 0.77).pretty_print_accuracy_top_n()
    # Hybrid(data).fit(True, False, 1.0).pretty_print_accuracy_top_n()
    # model = Hybrid(data).fit(False, False, 0.4)
    # model = Hybrid(data).fit(True, True, 1.0)

    # user_id = 330976
    # print(f"\nGetting recommendations for user {user_id}")

    # n = 15
    # likes = data.get_liked_movies_indices(user_id, True, "train")[:n]

    # print(f"User {user_id} previously liked:")
    # print(data.get_movie_from_indices(likes[:n]))

    # user_index = data.user_id_to_index[user_id]
    # recs = model.top_n(user_index, n)
    # print(f"Recommendations for user {user_id}")
    # print(data.get_movies_from_ids(recs))

    # model = Word2VecSession().fit(data)
    # model.pretty_print_accuracy_top_n()
    # recs = model.top_n(user_index, n)
    # print(f"Recommendations for user {user_id}")
    # print(data.get_movies_from_ids(recs))
