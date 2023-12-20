import numpy as np
from data import Data
from models import SGD, ALS, Content_Based, Hybrid, Word2VecSession
from utils import exp_range_float, exp_range_int


def cv(data: Data):
    sgd_params_space = {
        "n_factors": exp_range_int(5, 100, 6),
        "epochs": exp_range_int(5, 30, 6),
        "lr": exp_range_float(0.001, 0.1, 7),
        "reg": np.linspace(0.001, 0.005, 5),
        "batch_size": [1, 4, 8, 16],
        "lr_decay_factor": np.linspace(0.5, 0.99, 4),
    }
    SGD(data).gridsearch_cv("prediction", sgd_params_space, True)

    als_params_space = {
        "n_factors": exp_range_int(2, 100, 30),
        "epochs": exp_range_int(5, 50, 30),
        "reg": exp_range_float(0.001, 0.9, 35),
    }
    ALS(data).gridsearch_cv("prediction", als_params_space, True)

    cb_params_space = {
        "by_timestamp": [True, False],
        "biased": [True, False],
        "like_perc": np.linspace(0.01, 1.0, 100),
    }
    cb = Content_Based(data)
    cb_results = cb.gridsearch_cv("top_n", cb_params_space, False)
    cb.pretty_print_cv_results("top_n", cb_results)
    cb_results = [x for x in cb_results if x[8]["by_timestamp"]]
    cb.pretty_print_cv_results("top_n", cb_results)

    hb_params_space = cb_params_space
    hb = Hybrid(data)
    hb_results = hb.gridsearch_cv("top_n", hb_params_space, False)
    hb.pretty_print_cv_results("top_n", hb_results)
    hb_results = [x for x in hb_results if x[8]["by_timestamp"]]
    hb.pretty_print_cv_results("top_n", hb_results)

    w2v_params_space = {
        "vector_size": np.linspace(1, 100, 100, dtype=int),
        "window": np.linspace(1, 50, 50, dtype=int),
        "biased": [True, False],
    }
    Word2VecSession(data).gridsearch_cv("top_n", w2v_params_space, True)
