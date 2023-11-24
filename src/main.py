from data import Data
from utils import RandomSingleton
from non_pers import non_pers
from cf import cf
from cb import cb
from lightfm_model import lightfm_model

RandomSingleton.initialize(seed=42)


def main():
    lightfm_model()

    # data = Data(
    #     movies_path="data/u.item", ratings_path="data/u.data", ratings_test_size=0.25
    # )
    # cb(data)
    # cf(data)
    # non_pers(data)


if __name__ == "__main__":
    main()
