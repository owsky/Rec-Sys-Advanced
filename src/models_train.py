from scipy.sparse import coo_array
from models import MF, ALS_MR, ALS
from pyspark_model import pyspark_als


def models_train(train_set: coo_array, test_set: coo_array):
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
