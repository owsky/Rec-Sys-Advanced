from data import load_data, train_test_split
from models import MF


def main():
    data = load_data("data/u.data")
    train_set, test_set = train_test_split(data)


if __name__ == "__main__":
    main()
