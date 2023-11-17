from pandas import DataFrame
from scipy.sparse import coo_array
from data import RatingsReader, get_movie_info

from models import highest_rated, most_popular


def non_pers(ratings: coo_array, item_features: DataFrame, reader: RatingsReader):
    rec_id = highest_rated(ratings.tocsc(), reader.id_to_index(13, "user"))
    if rec_id:
        rec_movie = get_movie_info(item_features, rec_id)
        print(f"Movie recommendation for user 13: {rec_movie}")

    rec_id = most_popular(ratings.tocsc(), reader.id_to_index(13, "user"))
    if rec_id:
        rec_movie = get_movie_info(item_features, rec_id)
        print(f"Movie recommendation for user 13: {rec_movie}")
