from data import Data
from models import SGD, ALS, Content_Based, Hybrid, Word2VecSession
import numpy as np


def cv(data: Data):
    # sgd_params_space = {
    #     "n_factors": [5, 10, 20],
    #     "epochs": [10, 20, 30],
    #     "lr": np.arange(0.001, 0.01, 0.003),
    #     "batch_size": [1, 8, 16],
    #     "lr_decay_factor": [0.8, 0.9, 0.99],
    # }
    # SGD(data).gridsearch_cv("prediction", sgd_params_space)

    # als_params_space = {
    #     "n_factors": [2, 10, 20],
    #     "epochs": [10, 20, 30],
    #     "reg": np.arange(0.2, 1.0, 0.2),
    # }
    # ALS(data).gridsearch_cv("prediction", als_params_space)

    # cb_params_space = {
    #     "by_timestamp": [True, False],
    #     "biased": [True, False],
    #     "like_perc": np.arange(0.1, 1.0, 0.1),
    # }
    # Content_Based(data).gridsearch_cv("top_n", cb_params_space)

    # hb_params_space = cb_params_space
    # Hybrid(data).gridsearch_cv("top_n", hb_params_space)

    w2v_params_space = {
        "vector_size": np.arange(1, 100, 5),
        "window": np.arange(1, 20, 4),
        "biased": [True, False],
    }
    Word2VecSession(data).gridsearch_cv("top_n", w2v_params_space)
