from data import Data
from models import SGD, ALS_MR, ALS, User_Based, Item_Based, Ensemble


def cf(data: Data):
    # Stochastic Gradient Descent
    sgd = SGD(data).fit()
    sgd.pretty_print_accuracy_predictions()
    sgd.pretty_print_accuracy_top_n()

    # Alternating Least Squares
    als = ALS(data).fit(n_factors=2, epochs=49, reg=0.7368057796054976)
    als.pretty_print_accuracy_predictions()
    als.pretty_print_accuracy_top_n()

    # Map Reduce Alternating Least Squares using PySpark
    als_mr = ALS_MR(data).fit(n_factors=2, epochs=49, reg=0.7368057796054976)
    als_mr.pretty_print_accuracy_predictions()
    als_mr.pretty_print_accuracy_top_n()

    # Nearest Neighbors User Based
    nn_user = User_Based(data, similarity="cosine").fit()
    nn_user.pretty_print_accuracy_predictions()
    # Nearest Neighbors Item Based
    nn_item = Item_Based(data, similarity="cosine").fit()
    nn_item.pretty_print_accuracy_predictions()
    nn_item.pretty_print_accuracy_top_n()

    # Ensemble
    ens = Ensemble(data=data, sgd_model=sgd, als_model=als, nn_model=nn_item).fit()
    ens.pretty_print_accuracy_predictions()
    ens.pretty_print_accuracy_top_n()
