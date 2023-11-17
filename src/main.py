from data import load_ratings, load_item_features
from data import train_test_split
from utils import RandomSingleton
from models import Nearest_neighbors

RandomSingleton.initialize(seed=42)


def main():
    ratings = load_ratings("data/u.data")

    train, test = train_test_split(df=ratings, keep_ids=False)

    item_features = load_item_features("data/u.item")
    # item_features_reduced = item_features.to_numpy()[:, 5:]
    # user_features = load_user_features("data/u.user")

    model = Nearest_neighbors()
    model.fit(train.todense(), "user", "pearson")
    indices = model.get_recommendations(1, 5)
    print(indices)

    # sim = compute_similarity_matrix(item_features_reduced)
    # print(f"0th element: {item_features.to_numpy()[0]}")
    # indices = retrieve_top_k_for_item(sim, 0, 5)
    # for index in indices:
    #     print(item_features.to_numpy()[index])


if __name__ == "__main__":
    main()
