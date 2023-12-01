from data import Data
from models import SVD, ALS_MR, ALS, Neighborhood_Based, Ensemble


def cf(data: Data):
    # Singular Value Decomposition
    svd = SVD()
    svd.fit(data.train)
    print(
        f"SVD MAE: {svd.accuracy_mae(data.test)}, SVD RMSE: {svd.accuracy_rmse(data.test)}"
    )

    # Alternating Least Squares
    als = ALS()
    als.fit(train_set=data.train, n_factors=2, epochs=10, reg=0.8)
    print(
        f"ALS MAE:  {als.accuracy_mae(data.test)}, ALS RMSE:  {als.accuracy_rmse(data.test)}"
    )

    # Map Reduce Alternating Least Squares using PySpark
    als_mr = ALS_MR()
    als_mr.fit(data, n_factors=2, epochs=10, reg=0.8)
    print(
        f"ALS_MR MAE:  {als_mr.accuracy_mae(data.test)}, ALS_MR RMSE:  {als_mr.accuracy_rmse(data.test)}"
    )

    # Nearest Neighbors
    nn = Neighborhood_Based(kind="user", similarity="cosine").fit(data)
    print(
        f"User-based with Adjusted Cosine Similarity MAE: {nn.accuracy_mae(data.test)}, RMSE: {nn.accuracy_rmse(data.test)}"
    )

    # Ensemble
    ens = Ensemble(data=data, svd_model=svd, als_model=als, nn_model=nn)
    print(
        f"Ensemble MAE: {ens.accuracy_mae(data.test)}, Ensemble RMSE: {ens.accuracy_rmse(data.test)}"
    )
