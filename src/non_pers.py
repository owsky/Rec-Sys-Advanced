from data import Data
from models.non_personalized import Highest_Rated, Most_Popular


def non_pers(data: Data):
    hr_model = Highest_Rated(data).fit()
    hr_model.pretty_print_accuracy_top_n()

    mp_model = Most_Popular(data).fit()
    mp_model.pretty_print_accuracy_top_n()

    # user_id = 330976
    # user_index = data.user_id_to_index[user_id]

    # print(f"Recommending highest rated movies to user {user_id}:")
    # hr_movies = hr_model.top_n(user_index, 15)
    # print(data.get_movies_from_ids(hr_movies))

    # print(f"\nRecommending most popular movies to user {user_id}:")
    # mp_movies = mp_model.top_n(user_index, 15)
    # print(data.get_movies_from_ids(mp_movies), "\n")
