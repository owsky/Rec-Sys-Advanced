from data import load_data, train_test_split
from models import MF
from train import MF_cross_validate


def main():
    data = load_data("data/u.data")
    train_set, test_set = train_test_split(data)
    model = MF()
    MF_cross_validate(model, train_set, test_set)


if __name__ == "__main__":
    main()
