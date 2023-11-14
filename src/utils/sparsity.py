from scipy.sparse import coo_array


def sparsity(dataset: coo_array) -> float:
    """
    Compute the sparsity of the given dataset
    """
    total_number_elements = dataset.size
    non_zero_elements = dataset.count_nonzero()
    zero_elements = total_number_elements - non_zero_elements
    return zero_elements / total_number_elements
