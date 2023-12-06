import numpy as np
from data import Data
from utils import RandomSingleton
from non_pers import non_pers
from cf import cf
from cb import cb
from hb import hb

RandomSingleton.initialize(seed=42)


def main():
    data = Data(data_path="dataset/old/preprocessed/", ratings_test_size=0.3)
    # cf(data)
    non_pers(data)
    # cb(data)
    # hb(data)


if __name__ == "__main__":
    main()
