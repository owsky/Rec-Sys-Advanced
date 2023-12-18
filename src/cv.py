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
    SGD(data).gridsearch_cv("prediction", sgd_params_space)

    # +--------+--------+-----------------------------------------------------------------------------------------------------------------------------------+
    # |  MAE   |  RMSE  |                                                          Hyperparameters                                                          |
    # +========+========+===================================================================================================================================+
    # | 0.6826 | 0.9100 | {'n_factors': 5, 'epochs': 29, 'lr': 0.004641588833612777, 'reg': 0.005, 'batch_size': 16, 'lr_decay_factor': 0.6633333333333333} |
    # +--------+--------+-----------------------------------------------------------------------------------------------------------------------------------+
    # | 0.6835 | 0.9141 |         {'n_factors': 5, 'epochs': 29, 'lr': 0.004641588833612777, 'reg': 0.002, 'batch_size': 4, 'lr_decay_factor': 0.5}         |
    # +--------+--------+-----------------------------------------------------------------------------------------------------------------------------------+
    # | 0.6838 | 0.9136 |        {'n_factors': 5, 'epochs': 29, 'lr': 0.004641588833612777, 'reg': 0.003, 'batch_size': 1, 'lr_decay_factor': 0.99}         |
    # +--------+--------+-----------------------------------------------------------------------------------------------------------------------------------+
    # | 0.6840 | 0.9144 |        {'n_factors': 5, 'epochs': 29, 'lr': 0.004641588833612777, 'reg': 0.005, 'batch_size': 1, 'lr_decay_factor': 0.99}         |
    # +--------+--------+-----------------------------------------------------------------------------------------------------------------------------------+
    # | 0.6848 | 0.9153 |        {'n_factors': 5, 'epochs': 24, 'lr': 0.004641588833612777, 'reg': 0.003, 'batch_size': 16, 'lr_decay_factor': 0.99}        |
    # +--------+--------+-----------------------------------------------------------------------------------------------------------------------------------+

    als_params_space = {
        "n_factors": exp_range_int(2, 100, 30),
        "epochs": exp_range_int(5, 50, 30),
        "reg": exp_range_float(0.001, 0.9, 35),
    }

    ALS(data).gridsearch_cv("prediction", als_params_space)

    cb_params_space = {
        "by_timestamp": [True, False],
        "biased": [True, False],
        "like_perc": np.linspace(0.1, 1.0, 100),
    }
    Content_Based(data).gridsearch_cv("top_n", cb_params_space)

    hb_params_space = cb_params_space
    Hybrid(data).gridsearch_cv("top_n", hb_params_space)

    w2v_params_space = {
        "vector_size": exp_range_int(1, 100, 20),
        "window": exp_range_int(1, 2, 20),
        "biased": [True, False],
    }
    Word2VecSession(data).gridsearch_cv("top_n", w2v_params_space)
