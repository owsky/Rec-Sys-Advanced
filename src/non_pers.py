from data import Data
from models import Non_Personalized


def non_pers(data: Data):
    print("Fitting non personalized model")
    model = Non_Personalized()
    model.fit(data)
    print(model.accuracy("highest_rated"))
    print(model.accuracy("most_popular"))
    print("Recommending highest rated movies to user 13:")
    hr_movies = model.get_n_highest_rated(13)
    data.pretty_print_movies_df(hr_movies)

    print("Recommending most popular movies to user 13:")
    mp_movies = model.get_n_most_popular(13)
    data.pretty_print_movies_df(mp_movies)
