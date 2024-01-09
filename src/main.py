from data import Data
from utils import RandomSingleton
from non_pers import non_pers
from cf import cf
from cb import cb
from hb import hb
from cv import cv
import sys
import signal

RandomSingleton.initialize(seed=42)


def main():
    # Handle graceful shutdown when the user terminates the CLI execution
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

    data = Data(data_path="dataset/preprocessed/", ratings_test_size=0.3)

    cv(data)
    # cf(data)
    # non_pers(data)
    # cb(data)
    # hb(data)


if __name__ == "__main__":
    main()
