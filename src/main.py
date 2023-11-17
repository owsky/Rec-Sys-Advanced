import numpy as np
from data import get_movie_info, load_item_features, RatingsReader
from utils import RandomSingleton
from models import Nearest_neighbors, most_popular, highest_rated
from scipy.sparse import csc_array

RandomSingleton.initialize(seed=42)


def main():
    reader = RatingsReader()
    train, test = reader.load_ratings("data/u.data", test_size=0.25)
    item_features = load_item_features("data/u.item")
    # item_features_reduced = item_features.to_numpy()[:, 5:]
    # user_features = load_user_features("data/u.user")

    # model = Nearest_neighbors()
    # model.fit(train.todense(), "user", "pearson")
    # indices = model.get_recommendations(1, 5)
    # print(indices)

    # sim = compute_similarity_matrix(item_features_reduced)
    # print(f"0th element: {item_features.to_numpy()[0]}")
    # indices = retrieve_top_k_for_item(sim, 0, 5)
    # for index in indices:
    #     print(item_features.to_numpy()[index])

    rec_id = highest_rated(train.tocsc(), reader.id_to_index(13, "user"))
    if rec_id:
        rec_movie = get_movie_info(item_features, rec_id)
        print(f"Movie recommendation for user 13: {rec_movie}")


if __name__ == "__main__":
    main()
