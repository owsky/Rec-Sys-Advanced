from pandas import DataFrame
from models import Content_Based


def cb(item_features_df: DataFrame):
    model = Content_Based()
    model.fit(item_features_df)
    print("Getting recommendations for items similar to:")
    print(model.retrieve_movies(201))
    print(model.get_recommendations(201))
