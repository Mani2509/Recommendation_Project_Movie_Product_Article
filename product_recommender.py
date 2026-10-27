
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "products" / "styles.csv"
IMAGE_DIR = BASE_DIR / "data" / "products" / "images"


def load_products():
    """Load product details from CSV."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Product dataset not found: {DATA_FILE}")

    # Skip malformed CSV rows for now.
    products = pd.read_csv(DATA_FILE, on_bad_lines="skip")

    required_columns = [
        "id",
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "productDisplayName",
    ]

    missing = [
        column for column in required_columns
        if column not in products.columns
    ]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    products = products.dropna(
        subset=["id", "productDisplayName"]
    ).copy()

    products["id"] = products["id"].astype(int)
    products = products.drop_duplicates(subset=["id"])
    products = products.reset_index(drop=True)

    return products


def build_product_model(products):
    """Build a TF-IDF model using product attributes."""
    feature_columns = [
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "productDisplayName",
    ]

    products = products.copy()

    products["features"] = (
        products[feature_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    product_matrix = vectorizer.fit_transform(products["features"])

    return products, vectorizer, product_matrix


def recommend_products(product_id, products, product_matrix, top_n=5):
    """Recommend products similar to a selected product."""
    matching = products.index[products["id"] == int(product_id)].tolist()

    if not matching:
        raise ValueError(f"Product ID {product_id} was not found.")

    selected_index = matching[0]

    scores = cosine_similarity(
        product_matrix[selected_index],
        product_matrix
    ).flatten()

    scores[selected_index] = -1

    recommended_indices = scores.argsort()[::-1][:top_n]

    results = products.iloc[recommended_indices].copy()
    results["similarity_score"] = scores[recommended_indices]

    return results


def get_product_image(product_id):
    """Return the local image path if it exists."""
    image_path = IMAGE_DIR / f"{int(product_id)}.jpg"

    if image_path.exists():
        return str(image_path)

    return None


if __name__ == "__main__":
    products = load_products()
    products, vectorizer, product_matrix = build_product_model(products)

    print(f"Products loaded: {len(products):,}")

    first_product = products.iloc[0]
    print("\nSelected product:")
    print(first_product["productDisplayName"])

    recommendations = recommend_products(
        first_product["id"],
        products,
        product_matrix,
        top_n=5
    )

    print("\nRecommended products:")
    print(
        recommendations[
            ["id", "productDisplayName", "baseColour", "similarity_score"]
        ].to_string(index=False)
    )
