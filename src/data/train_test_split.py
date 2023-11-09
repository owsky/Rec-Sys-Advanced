from typing import Callable
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from scipy.sparse import coo_array


# Partition the dataframe according to the "by" function parameter
def _partition_df(df: DataFrameGroupBy, by: Callable) -> pd.DataFrame:
    df_applied: pd.DataFrame = df.apply(by)  # type: ignore because Python types suck
    return df_applied.reset_index(drop=True)


# Convert the dataframe into a user-item ratings matrix while preserving all item_ids
def _df_to_ratings(df: pd.DataFrame, total_items: int) -> pd.DataFrame:
    return df.pivot_table(
        values="rating", index="user_id", columns="item_id", fill_value=0
    ).reindex(columns=range(1, total_items + 1), fill_value=0)


# Split the input dataframe into two sparse arrays for train and test with given ratio
def train_test_split(df: pd.DataFrame, ratio=0.75) -> tuple[coo_array, coo_array]:
    # Sort users' ratings in-place by timestamp
    df.sort_values(["user_id", "timestamp"], inplace=True)

    # Group ratings by user
    data_grouped = df.groupby("user_id")

    # Partition according to ratio
    train_set = _partition_df(data_grouped, lambda x: x.head(int(len(x) * ratio)))
    test_set = _partition_df(data_grouped, lambda x: x.tail(int(len(x) * (1 - ratio))))

    # Get total number of unique items
    total_items = df["item_id"].nunique()

    # Convert into user-item matrices
    train_set_matrix = _df_to_ratings(train_set, total_items)
    test_set_matrix = _df_to_ratings(test_set, total_items)

    return coo_array(train_set_matrix), coo_array(test_set_matrix)
