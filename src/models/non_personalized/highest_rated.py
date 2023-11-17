import numpy as np
from scipy.sparse import csc_array, csr_array


def highest_rated(ratings: csc_array, user: int) -> int | None:
    user_row = csr_array(ratings.getrow(user))
    # Identify unrated items for the specified user
    unrated_items = np.where(user_row.toarray() == 0)[1]

    # If all items are rated, return None or handle it based on your requirements
    if len(unrated_items) == 0:
        return None

    # Get the ratings for unrated items
    unrated_item_ratings = ratings[:, unrated_items].toarray()

    # Find the index of the highest rated unrated item
    highest_rated_index = np.unravel_index(
        np.argmax(unrated_item_ratings), unrated_item_ratings.shape
    )[1]

    # Return the highest rated unrated item
    return unrated_items[highest_rated_index]
