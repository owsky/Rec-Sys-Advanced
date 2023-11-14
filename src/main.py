from data import train_test_split, load_data
from models import (
    MF,
    ALS_MR,
    ALS,
    cv_hyper_mf_helper,
    cv_hyper_als_helper,
    cv_hyper_als_mr_helper,
)
from pyspark_model import pyspark_als
from utils import RandomSingleton


def main():
    RandomSingleton.initialize(seed=42)
    data = load_data("data/u.data")
    train_set, test_set = train_test_split(data)

    # Cross validation
    cv_hyper_mf_helper(train_set, test_set)
    cv_hyper_als_helper(train_set, test_set)
    cv_hyper_als_mr_helper(train_set, test_set)

    # Matrix Factorization
    mf = MF()
    mf.fit(train_set)
    print(f"MF MAE: {mf.accuracy_mae(test_set)}, MF RMSE: {mf.accuracy_rmse(test_set)}")

    # Alternating Least Squares
    als = ALS()
    als.fit(train_set=train_set, n_factors=2, epochs=10, reg=0.8)
    print(
        f"ALS MAE:  {als.accuracy_mae(test_set)}, ALS RMSE:  {als.accuracy_rmse(test_set)}"
    )

    # Map Reduce Alternating Least Squares using PySpark
    als_mr = ALS_MR()
    als_mr.fit(train_set=train_set, n_factors=2, epochs=10, reg=0.8)
    print(
        f"ALS_MR MAE:  {als_mr.accuracy_mae(test_set)}, ALS RMSE:  {als_mr.accuracy_rmse(test_set)}"
    )

    # Bundled Alternating Least Squares from PySpark
    pyspark_als(train_set, test_set)


if __name__ == "__main__":
    main()
