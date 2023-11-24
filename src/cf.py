import numpy as np
from data import Data
from models import MF, ALS_MR, ALS, Nearest_Neighbors
from pyspark_model import pyspark_als
from collections import defaultdict
from surprise import Dataset, KNNBaseline, KNNBasic
from surprise.model_selection import KFold
from surprise.similarities import pearson


def cf(data: Data):
    # Matrix Factorization
    mf = MF()
    mf.fit(data.train)
    print(
        f"MF MAE: {mf.accuracy_mae(data.test)}, MF RMSE: {mf.accuracy_rmse(data.test)}"
    )

    # Alternating Least Squares
    als = ALS()
    als.fit(train_set=data.train, n_factors=2, epochs=10, reg=0.8)
    print(
        f"ALS MAE:  {als.accuracy_mae(data.test)}, ALS RMSE:  {als.accuracy_rmse(data.test)}"
    )

    # Map Reduce Alternating Least Squares using PySpark
    als_mr = ALS_MR()
    als_mr.fit(train_set=data.train, n_factors=2, epochs=10, reg=0.8)
    print(
        f"ALS_MR MAE:  {als_mr.accuracy_mae(data.test)}, ALS RMSE:  {als_mr.accuracy_rmse(data.test)}"
    )

    # Bundled Alternating Least Squares from PySpark
    pyspark_als(data.train, data.test)

    # Nearest Neighbors
    nn = Nearest_Neighbors()
    nn.fit(data, "user", "cosine")
    print(f"User-based MAE: {nn.accuracy_mae()}, RMSE: {nn.accuracy_rmse()}")
    nn2 = Nearest_Neighbors()
    nn2.fit(data, "item", "cosine")
    print(f"Item-based MAE: {nn2.accuracy_mae()}, RMSE: {nn2.accuracy_rmse()}")
