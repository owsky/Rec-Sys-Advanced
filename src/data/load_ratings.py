import pandas as pd


# Read data from file into Pandas DataFrame
def load_ratings(
    ratings_file_path: str,
):
    ratings_df = pd.read_csv(
        ratings_file_path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    return ratings_df
