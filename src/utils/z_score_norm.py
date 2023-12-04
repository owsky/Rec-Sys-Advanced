import numpy as np
from scipy.sparse import coo_array


def z_score_norm(ratings: coo_array):
    # Initialize an empty COO matrix for normalized values
    normalized_data = []
    normalized_rows = []
    normalized_cols = []

    # Iterate through each row
    for i in range(ratings.shape[0]):
        # Extract non-zero elements in the row
        row_elements = ratings.data[ratings.row == i]
        if row_elements.any():
            # Compute mean and standard deviation excluding zero values
            mean_value = np.mean(row_elements)
            std_value = np.std(row_elements)

            # Normalize non-zero elements using z-score
            if std_value == 0:
                std_value = 1
            normalized_values = (row_elements - mean_value) / std_value

            # Append normalized values to the result COO matrix
            normalized_data.extend(normalized_values)
            normalized_rows.extend([i] * len(normalized_values))
            normalized_cols.extend(ratings.col[ratings.row == i])

    normalized_coo_matrix = coo_array(
        (normalized_data, (normalized_rows, normalized_cols)), shape=ratings.shape
    )

    return normalized_coo_matrix
