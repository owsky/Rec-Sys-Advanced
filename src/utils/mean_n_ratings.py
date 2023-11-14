from scipy.sparse import coo_array
import numpy as np


def mean_n_ratings(dataset: coo_array):
    """
    Count the mean of non-zero entries by row and by column
    """
    rows = np.mean(np.diff(dataset.tocsr().indptr))
    cols = np.mean(np.diff(dataset.tocsc().indptr))
    return rows, cols
