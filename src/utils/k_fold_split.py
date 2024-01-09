from math import ceil
from scipy.sparse import coo_array
import numpy as np


def k_fold_split(train_dataset: coo_array):
    split_ratio = 0.8
    num_rows = train_dataset.shape[0]
    split_index = ceil(num_rows * split_ratio)

    # Get the row indices for the first and second splits
    rows_first_split = np.arange(split_index)
    rows_second_split = np.arange(split_index, num_rows)

    # Extract data, row indices, and column indices for both splits
    data_first_split = train_dataset.data[train_dataset.row < split_index]
    row_first_split = train_dataset.row[train_dataset.row < split_index]
    col_first_split = train_dataset.col[train_dataset.row < split_index]

    data_second_split = train_dataset.data[train_dataset.row >= split_index]
    row_second_split = train_dataset.row[train_dataset.row >= split_index]
    col_second_split = train_dataset.col[train_dataset.row >= split_index]

    # Create coo_matrix for both splits
    first_split_matrix = coo_array(
        (data_first_split, (row_first_split, col_first_split)),
        shape=train_dataset.shape,
    )
    second_split_matrix = coo_array(
        (data_second_split, (row_second_split, col_second_split)),
        shape=train_dataset.shape,
    )

    return first_split_matrix, second_split_matrix
