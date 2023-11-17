import pandas as pd


def load_item_features(items_file_path: str):
    data = pd.read_csv(
        items_file_path,
        sep="|",
        header=None,
        names=[
            "movie_id",
            "movie_title",
            "release_date",
            "video_release_date",
            "IMDb_URL",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ],
        encoding="iso-8859-1",
        on_bad_lines="warn",
    )
    data = data.drop(["video_release_date", "IMDb_URL"], axis=1)
    return data
