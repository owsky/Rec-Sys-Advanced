from models import cv_hyper_mf_helper, cv_hyper_als_helper, cv_hyper_als_mr_helper
from data import train_test_split


def cv(ratings):
    train_set, test_set = train_test_split(ratings)
    cv_hyper_mf_helper(train_set, test_set)
    cv_hyper_als_helper(train_set, test_set)
    cv_hyper_als_mr_helper(train_set, test_set)
