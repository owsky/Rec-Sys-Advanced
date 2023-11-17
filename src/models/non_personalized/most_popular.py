import numpy as np
from scipy.sparse import csc_array, csr_array


def most_popular(ratings: csc_array, user: int):
    user_row = csr_array(ratings.getrow(user))
    # Identify unrated items for the specified user
    unrated_items = np.where(user_row.toarray() == 0)[1]

    # If all items are rated, return None or handle it based on your requirements
    if len(unrated_items) == 0:
        return None

    # Get the popularity scores for unrated items
    popularity_scores = np.array(ratings[:, unrated_items].getnnz(axis=0))

    # Find the index of the most popular unrated item
    most_popular_index = np.argmax(popularity_scores)

    # Return the most popular unrated item
    return unrated_items[most_popular_index]
