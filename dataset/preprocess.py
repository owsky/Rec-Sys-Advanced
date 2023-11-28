import re
import numpy as np
import pandas as pd
import os

script_directory = os.path.dirname(os.path.abspath(__file__))


# Read small datasets into memory
small_movies_df = pd.read_csv(os.path.join(script_directory, "movies.csv"))


# Function to remove 'IMAX' from genres column
def remove_imax(genre_string):
    genres_list = genre_string.split("|")
    genres_list = [genre for genre in genres_list if genre != "IMAX"]
    return "|".join(genres_list)


# Apply the function to the 'genres' column
small_movies_df["genres"] = small_movies_df["genres"].apply(remove_imax)

# Read full datasets into memory
full_genome_scores_df = pd.read_csv(os.path.join(script_directory, "genome-scores.csv"))
full_genome_tags_df = pd.read_csv(os.path.join(script_directory, "genome-tags.csv"))


full_genome_tags_df["tag"] = full_genome_tags_df[
    "tag"
].str.lower()  # Making all tags lowercased for uniform format


for index, row in full_genome_tags_df.iterrows():
    tag = row.tag  # .split()  ## splitting words for spell check

    correct_tag = re.sub(
        r" \([^)]*\)", "", tag
    )  ## Removing all parenthesis and its contents, including the whitespace before

    if (
        "based" in correct_tag
    ):  ## This is necessary because it is a common tag and avoids the other if statements downstream
        full_genome_tags_df.loc[index, "tag"] = correct_tag  # type: ignore
        continue

    if (
        "-" in correct_tag
    ):  ## This is to keep "sci-fi" from being removed in the next if statement
        full_genome_tags_df.loc[index, "tag"] = correct_tag  # type: ignore
        continue

    if re.findall(r"\b\w{2}\b", correct_tag):
        full_genome_tags_df.loc[
            index, "tag"  # type: ignore
        ] = (
            np.NaN
        )  ## Replacing two-letter words; Need to maintain index ordering, will delete NaNs later

    elif re.findall(r"\b\w{1}\b", correct_tag):
        # Replacing one-letter words
        full_genome_tags_df.loc[index, "tag"] = np.NaN  # type: ignore

    elif (
        tag == correct_tag
    ):  ## This is for better performance since replacing significantly slows the process
        continue

    else:
        # Saves the corrected tag
        full_genome_tags_df.loc[index, "tag"] = correct_tag  # type: ignore
        pass

# Dropping all tags with words that are lower than two letters or less
full_genome_tags_df = full_genome_tags_df.dropna()


full_genome = pd.merge(
    full_genome_tags_df, full_genome_scores_df, on="tagId", how="inner"
)

# Assuming df1 is your first dataframe and df2 is your second dataframe
# Merge dataframes on 'movieId'
merged_df = pd.merge(small_movies_df, full_genome, on="movieId")

# Sort by 'relevance' in descending order
merged_df = merged_df.sort_values(by="relevance", ascending=False)

# Group by 'movieId' and take the top 50 rows for each group
top_50_tags_per_movie = merged_df.groupby("movieId").head(50)

# If needed, you can drop the 'relevance' column after obtaining the top 50 tags
# top_50_tags_per_movie = top_50_tags_per_movie.drop(columns="relevance")
tags = top_50_tags_per_movie[["tagId", "movieId", "tag"]].drop_duplicates()
tags.to_csv(os.path.join(script_directory, "tags.csv"), index=False)
small_movies_df.to_csv(
    os.path.join(script_directory, "movies_filtered.csv"), index=False
)
