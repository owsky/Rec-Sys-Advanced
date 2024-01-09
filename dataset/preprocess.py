import zipfile
import re
import numpy as np
import pandas as pd
import os

script_directory = os.path.dirname(os.path.abspath(__file__))


def unzip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_path)


# Unzip the ml-latest and ml-latest-small datasets
unzip_file(os.path.join(script_directory, "ml-latest-small.zip"), script_directory)
unzip_file(os.path.join(script_directory, "ml-latest.zip"), script_directory)

# Helper functions for constructing the input paths
make_path_full = lambda path: os.path.join(script_directory, "ml-latest", path)
make_path_small = lambda path: os.path.join(script_directory, "ml-latest-small", path)

# Check if output path exists, create it otherwise
output_dir = os.path.join(script_directory, "preprocessed")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
make_output_path = lambda path: os.path.join(output_dir, path)

# Loading full datasets
ratings_full = pd.read_csv(make_path_full("ratings.csv"))
genome_scores = pd.read_csv(make_path_full("genome-scores.csv"))
genome_tags = pd.read_csv(make_path_full("genome-tags.csv"))

# Loading small datasets
ratings_small = pd.read_csv(make_path_small("ratings.csv"))
tags_small = pd.read_csv(make_path_small("tags.csv"))
movies_small = pd.read_csv(make_path_small("movies.csv"))


def remove_imax(genre_string):
    genres_list = genre_string.split("|")
    genres_list = [genre for genre in genres_list if genre != "IMAX"]
    return "|".join(genres_list)


movies_small["genres"] = movies_small["genres"].apply(remove_imax)

genome_tags["tag"] = genome_tags["tag"].str.lower()
for index, row in genome_tags.iterrows():
    tag = row.tag

    # Removing all parenthesis and its contents, including the whitespace before
    correct_tag = re.sub(r" \([^)]*\)", "", tag)

    # This is necessary because it is a common tag and avoids the other if statements downstream
    if "based" in correct_tag:
        genome_tags.loc[index, "tag"] = correct_tag  # type: ignore
        continue

    # This is to keep "sci-fi" from being removed in the next if statement
    if "-" in correct_tag:
        genome_tags.loc[index, "tag"] = correct_tag  # type: ignore
        continue

    # Replacing two-letter words; Need to maintain index ordering, will delete NaNs later
    if re.findall(r"\b\w{2}\b", correct_tag):
        genome_tags.loc[index, "tag"] = np.NaN  # type: ignore

    # Replacing one-letter words
    elif re.findall(r"\b\w{1}\b", correct_tag):
        genome_tags.loc[index, "tag"] = np.NaN  # type: ignore

    # This is for better performance since replacing significantly slows the process
    elif tag == correct_tag:
        continue

    # Saves the corrected tag
    else:
        genome_tags.loc[index, "tag"] = correct_tag  # type: ignore
        pass
genome_tags = genome_tags.dropna()

# Remove tag relevance for movies that are not to be considered
genome_scores = genome_scores[genome_scores["movieId"].isin(tags_small["movieId"])]

# Merge the tags with their scores obtaining on each row a pair tag, movie and the relevance of the tag
full_genome = pd.merge(genome_tags, genome_scores, on="tagId", how="inner")

# Inner merge the movies dataframe with the tags dataframe, thus removing movies without tags
merged_df = pd.merge(movies_small, full_genome, on="movieId", how="inner")

# Sort by relevance in descending order
merged_df.sort_values(by="relevance", ascending=False, inplace=True)

# Group by movieId and take the top 50 tags for each movie
merged_df = merged_df.groupby("movieId").head(50)

# Retrieve surviving tags and save the dataframe to file system
tags = merged_df[["tagId", "movieId", "tag"]]
tags.to_csv(make_output_path("tags.csv"), index=False)

# Obtain the unique movie ids of the movies which survived the trimming
surviving_movie_ids = merged_df["movieId"].unique()

# Filter the movies dataframe with the surviving movie ids and save the dataframe to file system
movies_filtered = movies_small[movies_small["movieId"].isin(surviving_movie_ids)]
movies_filtered.to_csv(make_output_path("movies.csv"), index=False)

# Select all unique user ids from the small ratings dataset and add an arbitrary amount of unique
# user ids from the full ratings dataset, such that the final result will contain ~100k ratings
user_ids = np.concatenate(
    (ratings_small["userId"].unique(), ratings_full["userId"].unique()[-1420:])
)

# Filter the full ratings dataset by removing movies that were trimmed and users who are not present in the
# small ratings dataset, then save the dataframe to file system
ratings_filtered = ratings_full[
    (ratings_full["movieId"].isin(surviving_movie_ids))
    & (ratings_full["userId"].isin(user_ids))
]
ratings_filtered.to_csv(make_output_path("ratings.csv"), index=False)
