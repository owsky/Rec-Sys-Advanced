from pandas import DataFrame

all_genres = [
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
]


def get_movie_info(movie_df: DataFrame, movie_id: int) -> tuple[str, list[str]]:
    """
    Given the movies DataFrame and a movie ID, returns the title and genres of the movie
    """
    rec_movie = movie_df.loc[movie_df["movie_id"] == movie_id].values[0]
    title = rec_movie[1]
    genres = [s for s, flag in zip(all_genres, rec_movie[3:]) if flag]
    return title, genres
