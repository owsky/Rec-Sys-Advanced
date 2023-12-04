from data import Data
from models.non_personalized import Highest_Rated, Most_Popular


def non_pers(data: Data):
    print("\nNon personalized recommender")
    hr_model = Highest_Rated().fit(data)

    print("Highest rated recommender accuracy:")
    print(
        "Precision@k: {:.3f}, Recall@k: {:.3f}, F1: {:.3f}, "
        "Average Reciprocal Hit Rank: {:.3f}, "
        "Normalized Discounted Cumulative Gain: {:.3f}\n".format(
            *hr_model.accuracy_top_n()
        )
    )

    mp_model = Most_Popular().fit(data)
    print("Most popular recommender accuracy:")
    print(
        "Precision@k: {:.3f}, Recall@k: {:.3f}, F1: {:.3f}, "
        "Average Reciprocal Hit Rank: {:.3f}, "
        "Normalized Discounted Cumulative Gain: {:.3f}\n".format(
            *mp_model.accuracy_top_n()
        )
    )

    user_id = 1

    print(f"Recommending highest rated movies to user {user_id}:")
    hr_movies = hr_model.top_n(user_id, 10)
    print(data.get_movies_from_ids(hr_movies))

    print(f"\nRecommending most popular movies to user {user_id}:")
    mp_movies = mp_model.top_n(user_id, 10)
    print(data.get_movies_from_ids(mp_movies), "\n")
