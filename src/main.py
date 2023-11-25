from data import Data
from utils import RandomSingleton
from non_pers import non_pers
from cf import cf
from cb import cb
from lightfm_model import lightfm_model

RandomSingleton.initialize(seed=42)


def main():
    data = Data(
        movies_path="data/movies.csv",
        tags_path="data/tags.csv",
        ratings_path="data/ratings.csv",
        ratings_test_size=0.25,
    )
    cf(data)
    non_pers(data)


if __name__ == "__main__":
    main()
