import re
import numpy as np
import pandas as pd
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
make_path = lambda path: os.path.join(script_directory, path)

ratings = pd.read_csv(make_path("ratings.csv"))
tags = pd.read_csv(make_path("tags.csv"))
movies = pd.read_csv(make_path("movies.csv"))
genome_scores = pd.read_csv(make_path("genome-scores.csv"))
genome_tags = pd.read_csv(make_path("genome-tags.csv"))


# Function to remove 'IMAX' from genres column
def remove_imax(genre_string):
    genres_list = genre_string.split("|")
    genres_list = [genre for genre in genres_list if genre != "IMAX"]
    return "|".join(genres_list)


movies["genres"] = movies["genres"].apply(remove_imax)
genome_tags["tag"] = genome_tags["tag"].str.lower()

# Removing unimportant tags
for index, row in genome_tags.iterrows():
    tag = row.tag  # .split()  ## splitting words for spell check

    ## Removing all parenthesis and its contents, including the whitespace before
    correct_tag = re.sub(r" \([^)]*\)", "", tag)

    if (
        "based" in correct_tag
    ):  ## This is necessary because it is a common tag and avoids the other if statements downstream
        genome_tags.loc[index, "tag"] = correct_tag  # type: ignore
        continue

    if (
        "-" in correct_tag
    ):  ## This is to keep "sci-fi" from being removed in the next if statement
        genome_tags.loc[index, "tag"] = correct_tag  # type: ignore
        continue

    if re.findall(r"\b\w{2}\b", correct_tag):
        ## Replacing two-letter words; Need to maintain index ordering, will delete NaNs later
        genome_tags.loc[index, "tag"] = np.NaN  # type: ignore

    elif re.findall(r"\b\w{1}\b", correct_tag):
        # Replacing one-letter words
        genome_tags.loc[index, "tag"] = np.NaN  # type: ignore

    elif (
        tag == correct_tag
    ):  ## This is for better performance since replacing significantly slows the process
        continue

    else:
        # Saves the corrected tag
        genome_tags.loc[index, "tag"] = correct_tag  # type: ignore
        pass
# Dropping all tags with words that are lower than two letters or less
genome_tags = genome_tags.dropna()

tags_scores_df = pd.merge(genome_tags, genome_scores, on="tagId", how="inner")
merged_df = movies.merge(tags_scores_df, on="movieId").sort_values(
    by="relevance", ascending=False
)

merged_df = merged_df[merged_df["relevance"] >= 0.55].groupby("movieId").head(50)

tag_counts = (
    merged_df.groupby("movieId")["tag"]
    .count()
    .reset_index()
    .sort_values(by="tag", ascending=True)
)

filtered_tag_counts = tag_counts[tag_counts["tag"] < 15]
to_be_removed = filtered_tag_counts["movieId"].unique()
merged_df = merged_df[~merged_df["movieId"].isin(to_be_removed)]

tags = merged_df[["tagId", "movieId", "tag"]].drop_duplicates()
tags.to_csv(os.path.join(script_directory, "preprocessed", "tags.csv"), index=False)

surviving_movie_ids = list(
    set(merged_df["movieId"].unique()).intersection(set(ratings["movieId"].unique()))
)

surviving_user_ids = ratings["userId"].unique()[:1000]

ratings_filtered = ratings[
    (ratings["movieId"].isin(surviving_movie_ids))
    & (ratings["userId"].isin(surviving_user_ids))
]


movies_filtered = movies[movies["movieId"].isin(surviving_movie_ids)]

ratings_filtered = ratings_filtered[ratings_filtered["rating"] % 1 == 0]
ratings_filtered["rating"] = ratings_filtered["rating"].astype(int)

ratings_count = ratings_filtered.groupby(by="userId")["rating"].count().reset_index()
to_keep = ratings_count[ratings_count["rating"] >= 30]["userId"].unique()
ratings_filtered = ratings_filtered[ratings_filtered["userId"].isin(to_keep)]

ratings_filtered.to_csv(
    os.path.join(script_directory, "preprocessed", "ratings.csv"), index=False
)

surviving_movie_ids = ratings_filtered["movieId"].unique()
movies_filtered = movies[movies["movieId"].isin(surviving_movie_ids)]
movies_filtered.to_csv(
    os.path.join(script_directory, "preprocessed", "movies.csv"), index=False
)
