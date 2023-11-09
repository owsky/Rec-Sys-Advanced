import pandas as pd


# Read data from file into Pandas DataFrame
def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    return data
