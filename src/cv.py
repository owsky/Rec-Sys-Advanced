from models import cv_hyper_mf_helper, cv_hyper_als_helper, cv_hyper_als_mr_helper
from scipy.sparse import coo_array


def cv(train_set: coo_array, test_set: coo_array):
    cv_hyper_mf_helper(train_set, test_set)
    cv_hyper_als_helper(train_set, test_set)
    cv_hyper_als_mr_helper(train_set, test_set)
