from data import load_data, train_test_split
from models import MF
from train import MF_cross_validate
from utils import RandomSingleton, check_cold_start, simulate_cold_start
from scipy.sparse import coo_array


def main():
    data = load_data("data/u.data")
    trainset, testset = train_test_split(data)
    RandomSingleton.initialize(seed=42)
    trainset = simulate_cold_start(trainset)
    cold_users, cold_items = check_cold_start(trainset)
    print(f"There are {cold_users} cold users and {cold_items} cold items")


if __name__ == "__main__":
    main()
