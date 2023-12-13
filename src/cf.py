from data import Data
from models import SGD, ALS_MR, ALS, User_Based, Ensemble


def cf(data: Data):
    # Stochastic Gradient Descent
    sgd = SGD().fit(data)
    sgd.pretty_print_accuracy_predictions()
    sgd.pretty_print_accuracy_top_n()

    # Alternating Least Squares
    als = ALS().fit(data=data, n_factors=2, epochs=10, reg=0.8)
    als.pretty_print_accuracy_predictions()
    als.pretty_print_accuracy_top_n()

    # Map Reduce Alternating Least Squares using PySpark
    als_mr = ALS_MR().fit(data, n_factors=2, epochs=10, reg=0.8)
    als_mr.pretty_print_accuracy_predictions()
    als_mr.pretty_print_accuracy_top_n()

    # Nearest Neighbors
    nn = User_Based(kind="user", similarity="cosine").fit(data)
    nn.pretty_print_accuracy_predictions()
    nn.pretty_print_accuracy_top_n()

    # Ensemble
    ens = Ensemble(data=data, sgd_model=sgd, als_model=als, nn_model=nn).fit()
    ens.pretty_print_accuracy_predictions()
    ens.pretty_print_accuracy_top_n()
