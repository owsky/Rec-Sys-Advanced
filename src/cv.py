from data import Data
from models import SGD, ALS, Content_Based, Hybrid, Word2VecSession
import numpy as np


def cv(data: Data):
    sgd_params_space = {
        "n_factors": [5, 10, 15, 20, 50, 100],
        "epochs": [5, 10, 20, 30],
        "lr": np.arange(0.001, 0.1, 0.01),
        "reg": np.arange(0.001, 0.003, 0.001),
        "batch_size": [1, 4, 8, 16],
        "lr_decay_factor": [0.5, 0.8, 0.9, 0.99],
    }
    SGD(data).gridsearch_cv("prediction", sgd_params_space)


# Stochastic Gradient Descent CV results:
# +--------+--------+---------------------------------------------------------------------------------------+
# |  MAE   |  RMSE  |                                    Hyperparameters                                    |
# +========+========+=======================================================================================+
# | 0.6933 | 0.9233 | {'n_factors': 5, 'epochs': 10, 'lr': 0.011, 'batch_size': 8, 'lr_decay_factor': 0.5}  |
# +--------+--------+---------------------------------------------------------------------------------------+
# | 0.6934 | 0.9240 | {'n_factors': 5, 'epochs': 20, 'lr': 0.011, 'batch_size': 16, 'lr_decay_factor': 0.5} |
# +--------+--------+---------------------------------------------------------------------------------------+
# | 0.6937 | 0.9252 | {'n_factors': 5, 'epochs': 20, 'lr': 0.011, 'batch_size': 4, 'lr_decay_factor': 0.5}  |
# +--------+--------+---------------------------------------------------------------------------------------+
# | 0.6938 | 0.9278 | {'n_factors': 5, 'epochs': 20, 'lr': 0.011, 'batch_size': 8, 'lr_decay_factor': 0.9}  |
# +--------+--------+---------------------------------------------------------------------------------------+
# | 0.6943 | 0.9237 | {'n_factors': 5, 'epochs': 20, 'lr': 0.011, 'batch_size': 16, 'lr_decay_factor': 0.8} |
# +--------+--------+---------------------------------------------------------------------------------------+

# als_params_space = {
#     "n_factors": [2, 5, 10, 15, 20, 50, 100],
#     "epochs": [5, 10, 20, 30, 40],
#     "reg": np.arange(0.01, 0.8, 0.04),
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

# w2v_params_space = {
#     "vector_size": np.arange(1, 100, 5),
#     "window": np.arange(1, 20, 4),
#     "biased": [True, False],
# }
# Word2VecSession(data).gridsearch_cv("top_n", w2v_params_space)
