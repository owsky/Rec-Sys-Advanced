import re
import numpy as np
import pandas as pd
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
make_path = lambda path: os.path.join(script_directory, path)

# Loading full datasets
ratings_full = pd.read_csv(make_path("ratings_full.csv"))
tags_full = pd.read_csv(make_path("tags_full.csv"))
movies_full = pd.read_csv(make_path("movies_full.csv"))
genome_scores = pd.read_csv(make_path("genome-scores_full.csv"))
genome_tags = pd.read_csv(make_path("genome-tags_full.csv"))

# Loading small datasets
ratings_small = pd.read_csv(make_path("ratings_small.csv"))
tags_small = pd.read_csv(make_path("tags_small.csv"))
movies_small = pd.read_csv(make_path("movies_small.csv"))


# Function to remove 'IMAX' from genres column
def remove_imax(genre_string):
    genres_list = genre_string.split("|")
    genres_list = [genre for genre in genres_list if genre != "IMAX"]
    return "|".join(genres_list)


# Apply the function to the 'genres' column
movies_small["genres"] = movies_small["genres"].apply(remove_imax)

# Filter full tags
genome_tags["tag"] = genome_tags[
    "tag"
].str.lower()  # Making all tags lowercased for uniform format


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


full_genome = pd.merge(genome_tags, genome_scores, on="tagId", how="inner")

full_genome = full_genome[full_genome["movieId"].isin(tags_small["movieId"])]

merged_df = pd.merge(movies_small, full_genome, on="movieId")

# Sort by 'relevance' in descending order
merged_df = merged_df.sort_values(by="relevance", ascending=False)

# Group by 'movieId' and take the top 50 rows for each group
top_50_tags_per_movie = merged_df.groupby("movieId").head(50)
tags = top_50_tags_per_movie[["tagId", "movieId", "tag"]].drop_duplicates()
tags.to_csv(os.path.join(script_directory, "preprocessed", "tags.csv"), index=False)
movies_small.to_csv(
    os.path.join(script_directory, "preprocessed", "movies.csv"), index=False
)
ratings_small.to_csv(
    os.path.join(script_directory, "preprocessed", "ratings.csv"), index=False
)
