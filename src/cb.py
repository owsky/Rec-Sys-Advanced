from data import Data
from models import Content_Based
import numpy as np
from numpy.typing import NDArray


def f(user_index: int, model: Content_Based):
    ratings = model.train[user_index, :]
    if np.std(ratings) == 0:
        nonzer_indices = np.nonzero(ratings >= 3.0)[0]
        most_liked_indices = sorted(
            nonzer_indices, key=lambda index: ratings[index], reverse=True  # type: ignore
        )
    else:
        user_bias = model.data.average_user_rating[user_index]
        liked_indices = np.nonzero(ratings - user_bias >= 0)[0]
        most_liked_indices = sorted(
            liked_indices, key=lambda index: ratings[index], reverse=True  # type: ignore
        )
    movie_vectors: list[NDArray[np.float64]] = []
    # for most_liked_index in most_liked_indices:
    # movie_vectors.append(model._get_movie_vector(most_liked_index).todense())
    # most_liked_ids = [model.data.item_index_to_id[i] for i in most_liked_indices]
    print(f"Most liked movies for user {model.data.user_index_to_id[user_index]}")
    print(model.data.get_movie_from_indices(most_liked_indices))


def cb(data: Data):
    model = Content_Based(data)
    model.fit()

    # user_id = 31
    # user_index = data.user_id_to_index[user_id]
    # f(user_index, model)
    # print(data.get_movies_from_ids(model.get_top_n_recommendations(user_index)))

    print(model.accuracy_metrics())
