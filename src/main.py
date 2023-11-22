import numpy as np
from data import Data
from utils import RandomSingleton
from non_pers import non_pers
from scipy.sparse import coo_array, csr_array
from cf import cf

RandomSingleton.initialize(seed=42)


def main():
    data = Data(
        movies_path="data/u.item", ratings_path="data/u.data", ratings_test_size=0.25
    )
    cf(data)
    # TODO DEBUG NEIGHBORHOOD COSINE SIMILARITY

    # non_pers(data)


if __name__ == "__main__":
    main()
