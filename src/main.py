from data import load_item_features
from utils import RandomSingleton


RandomSingleton.initialize(seed=42)


def main():
    # reader = RatingsReader()
    # train, test = reader.load_ratings("data/u.data", test_size=0.25)
    # item_features_df = load_item_features("data/u.item")
    pass


if __name__ == "__main__":
    main()
