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
    """
    Word2Vec Session CV results:
    +-------------+---------------+----------+------------+--------+--------+--------+--------+----------------------------------------------------+
    |  Precision  |  Precision@k  |  Recall  |  Recall@k  |   F1   |  F1@k  |  ARHR  |  NDCG  |                  Hyperparameters                   |
    +=============+===============+==========+============+========+========+========+========+====================================================+
    |   0.0197    |    0.0184     |  0.0186  |   0.0180   | 0.0189 | 0.0181 | 0.0410 | 0.0734 | {'vector_size': 18, 'window': 2, 'biased': False}  |
    +-------------+---------------+----------+------------+--------+--------+--------+--------+----------------------------------------------------+
    |   0.0188    |    0.0177     |  0.0178  |   0.0173   | 0.0181 | 0.0174 | 0.0395 | 0.0672 | {'vector_size': 23, 'window': 2, 'biased': False}  |
    +-------------+---------------+----------+------------+--------+--------+--------+--------+----------------------------------------------------+
    |   0.0183    |    0.0173     |  0.0174  |   0.0170   | 0.0177 | 0.0171 | 0.0384 | 0.0653 | {'vector_size': 14, 'window': 2, 'biased': False}  |
    +-------------+---------------+----------+------------+--------+--------+--------+--------+----------------------------------------------------+
    |   0.0181    |    0.0171     |  0.0171  |   0.0167   | 0.0174 | 0.0168 | 0.0383 | 0.0658 | {'vector_size': 78, 'window': 2, 'biased': False}  |
    +-------------+---------------+----------+------------+--------+--------+--------+--------+----------------------------------------------------+
    |   0.0179    |    0.0169     |  0.0170  |   0.0166   | 0.0173 | 0.0167 | 0.0354 | 0.0638 | {'vector_size': 100, 'window': 2, 'biased': False} |
    +-------------+---------------+----------+------------+--------+--------+--------+--------+----------------------------------------------------+
    """
