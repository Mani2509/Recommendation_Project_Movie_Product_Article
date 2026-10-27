
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "movies"

RATINGS_FILE = DATA_DIR / "u.data"
MOVIES_FILE = DATA_DIR / "u.item"




def clean_movie_title(title):
    title = str(title).strip()

    # Move trailing article before the movie name,
    # preserving the release year.
    match = re.match(
        r"^(.*?)\s*,\s*(The|An|A)(\s+\(\d{4}\))?$",
        title,
        flags=re.IGNORECASE
    )

    if match:
        movie_name = match.group(1).strip()
        article = match.group(2)
        year = match.group(3) or ""

        title = f"{article} {movie_name}{year}"

    return title


def load_movie_data():
    ratings = pd.read_csv(
        RATINGS_FILE,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        MOVIES_FILE,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["movie_id", "title"]
    )

    return ratings, movies


def recommend_movies(user_id, top_n=10):
    ratings, movies = load_movie_data()

    if user_id not in ratings["user_id"].values:
        return pd.DataFrame(
            columns=["title", "predicted_rating"]
        )

    user_item = ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    )

    # User similarity based on co-rated movies
    similarity = cosine_similarity(
        user_item.fillna(0)
    )

    similarity_df = pd.DataFrame(
        similarity,
        index=user_item.index,
        columns=user_item.index
    )

    similar_users = similarity_df.loc[user_id].drop(
        labels=[user_id]
    ).sort_values(ascending=False)

    similar_users = similar_users[
        similar_users > 0
    ].head(20)

    if similar_users.empty:
        return pd.DataFrame(
            columns=["title", "predicted_rating"]
        )

    # Predict ratings using similar users' ratings
    weighted_ratings = (
        user_item.loc[similar_users.index]
        .fillna(0)
        .mul(similar_users, axis=0)
    )

    rated_mask = user_item.loc[similar_users.index].notna()

    weighted_sum = weighted_ratings.sum(axis=0)
    weight_sum = rated_mask.mul(
        similar_users, axis=0
    ).sum(axis=0)

    predictions = weighted_sum.div(
        weight_sum.replace(0, float("nan"))
    )

    # Exclude movies already rated by this user
    already_rated = user_item.loc[user_id].dropna().index
    predictions = predictions.drop(
        index=already_rated, errors="ignore"
    )

    results = predictions.dropna().sort_values(
        ascending=False
    ).head(top_n)

    output = results.rename(
        "predicted_rating"
    ).reset_index()

    output.columns = ["movie_id", "predicted_rating"]

    output = output.merge(
    movies, on="movie_id", how="left"
    )

    # Correct movie titles before returning results
    output["title"] = output["title"].apply(clean_movie_title)

    return output[["title", "predicted_rating"]]


def evaluate_movie_model(test_size=0.2, top_n=10):
    ratings, movies = load_movie_data()

    # Split ratings into train and test sets
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=42
    )

    # Build user-item matrix using training data only
    user_item = train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    )

    similarity = cosine_similarity(
        user_item.fillna(0)
    )

    similarity_df = pd.DataFrame(
        similarity,
        index=user_item.index,
        columns=user_item.index
    )

    actual_ratings = []
    predicted_ratings = []

    # Evaluate predictions for held-out ratings
    for row in test.itertuples(index=False):
        user_id = row.user_id
        movie_id = row.movie_id
        actual = row.rating

        if user_id not in user_item.index:
            continue

        if movie_id not in user_item.columns:
            continue

        similarities = similarity_df.loc[user_id].drop(
            labels=[user_id],
            errors="ignore"
        )

        rated_users = user_item[movie_id].dropna().index
        similarities = similarities.reindex(
            rated_users
        ).dropna()

        if similarities.empty:
            continue

        weighted = (
            user_item.loc[similarities.index, movie_id]
            * similarities
        )

        denominator = similarities.abs().sum()

        if denominator == 0:
            continue

        prediction = weighted.sum() / denominator

        prediction = max(1, min(5, prediction))

        actual_ratings.append(actual)
        predicted_ratings.append(prediction)

    if not actual_ratings:
        return {
            "RMSE": None,
            "MAE": None,
            "Precision": None,
            "Recall": None
        }

    rmse = mean_squared_error(
        actual_ratings,
        predicted_ratings
    ) ** 0.5

    mae = mean_absolute_error(
        actual_ratings,
        predicted_ratings
    )

    # Rating threshold: 4 or higher means relevant
    actual_relevant = [
        actual >= 4 for actual in actual_ratings
    ]

    predicted_relevant = [
        predicted >= 4 for predicted in predicted_ratings
    ]

    true_positive = sum(
        actual and predicted
        for actual, predicted in zip(
            actual_relevant, predicted_relevant
        )
    )

    predicted_positive = sum(predicted_relevant)
    actual_positive = sum(actual_relevant)

    precision = (
        true_positive / predicted_positive
        if predicted_positive else 0.0
    )

    recall = (
        true_positive / actual_positive
        if actual_positive else 0.0
    )

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4)
    }


