from data import Data
from models import cv_hyper_mf_helper, cv_hyper_als_helper, cv_hyper_als_mr_helper


def cv(data: Data):
    train_set = data.train
    test_set = data.test
    cv_hyper_mf_helper(train_set, test_set)
    # cv_hyper_als_helper(train_set, test_set)
    # cv_hyper_als_mr_helper(train_set, test_set)
