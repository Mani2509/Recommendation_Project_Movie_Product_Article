
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "articles"
    / "News_Category_Dataset_v3.json"
)


def load_articles():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Article dataset not found: {DATA_FILE}"
        )

    articles = pd.read_json(DATA_FILE, lines=True)

    articles = articles.dropna(
        subset=["headline", "short_description"]
    ).copy()

    articles["article_text"] = (
        articles["headline"].astype(str)
        + " "
        + articles["short_description"].astype(str)
    )

    articles = articles[
        articles["article_text"].str.strip() != ""
    ]

    return articles.reset_index(drop=True)


def build_article_model(articles):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(
        articles["article_text"]
    )

    return vectorizer, tfidf_matrix


def recommend_articles(
    article_index,
    articles,
    tfidf_matrix,
    top_n=5
):
    if article_index < 0 or article_index >= len(articles):
        raise ValueError("Invalid article index.")

    similarities = cosine_similarity(
        tfidf_matrix[article_index],
        tfidf_matrix
    ).flatten()

    # Exclude the selected article itself
    similarities[article_index] = -1

    recommended_indices = similarities.argsort()[
        -top_n:
    ][::-1]

    results = articles.iloc[
        recommended_indices
    ].copy()

    results["similarity_score"] = similarities[
        recommended_indices
    ]

    return results[
        [
            "category",
            "headline",
            "short_description",
            "link",
            "similarity_score"
        ]
    ].reset_index(drop=True)





def evaluate_article_model(
    articles,
    tfidf_matrix,
    sample_size=200,
    top_n=5,
    random_state=42
):
    """
    Proxy evaluation:
    Articles sharing the same category as the selected article
    are treated as relevant.

    These metrics are category-based proxy estimates, not
    human-judged relevance scores.
    """

    if "category" not in articles.columns:
        raise ValueError("Articles must contain a category column.")

    if len(articles) < 2:
        raise ValueError("At least two articles are required.")

    # Use articles with a known category.
    valid_indices = articles.index[
        articles["category"].notna()
    ].to_numpy()

    if len(valid_indices) < 2:
        raise ValueError("Not enough articles with valid categories.")

    rng = np.random.default_rng(random_state)

    # Limit evaluation to a reproducible random sample.
    sample_size = min(sample_size, len(valid_indices))
    sampled_indices = rng.choice(
        valid_indices,
        size=sample_size,
        replace=False
    )

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for article_index in sampled_indices:

        selected_category = articles.loc[
            article_index, "category"
        ]

        # All other articles with the same category
        # are considered relevant for this proxy evaluation.
        relevant_mask = (
            articles["category"].eq(selected_category).to_numpy(copy=True)
        )

        relevant_mask[article_index] = False

        total_relevant = int(relevant_mask.sum())

        if total_relevant == 0:
            continue

        # Get similarity scores for this article.
        similarity_scores = (
            tfidf_matrix[article_index]
            @ tfidf_matrix.T
        ).toarray().ravel()

        # Exclude the selected article itself.
        similarity_scores[article_index] = -np.inf

        # Find the top-N most similar articles.
        top_indices = np.argsort(
            similarity_scores
        )[-top_n:][::-1]

        # 1 = same category; 0 = different category.
        relevance = relevant_mask[top_indices].astype(int)

        relevant_recommended = int(relevance.sum())

        # Precision@K = relevant recommended / K.
        precision = relevant_recommended / top_n

        # Recall@K = relevant recommended / all relevant articles.
        recall = relevant_recommended / total_relevant

        # DCG: relevant articles appearing earlier get more weight.
        discounts = np.log2(
            np.arange(2, len(relevance) + 2)
        )

        dcg = float(
            np.sum(relevance / discounts)
        )

        # Ideal DCG for the available relevant articles.
        ideal_relevance = np.ones(
            min(top_n, total_relevant),
            dtype=float
        )

        ideal_discounts = np.log2(
            np.arange(2, len(ideal_relevance) + 2)
        )

        idcg = float(
            np.sum(ideal_relevance / ideal_discounts)
        )

        ndcg = dcg / idcg if idcg > 0 else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        ndcg_scores.append(ndcg)

    if not precision_scores:
        raise ValueError(
            "No articles could be evaluated using category labels."
        )

    return {
        "Precision@5": float(np.mean(precision_scores)),
        "Recall@5": float(np.mean(recall_scores)),
        "NDCG@5": float(np.mean(ndcg_scores)),
        "Evaluated Articles": len(precision_scores),
        "Evaluation Method": "Same-category proxy evaluation"
    }


if __name__ == "__main__":
    print("Loading articles...")

    articles = load_articles()

    print("Total articles:", len(articles))
    print("Building TF-IDF model...")

    vectorizer, tfidf_matrix = build_article_model(articles)

    print("Model ready!")

    selected_index = 0

    print("\nSelected article:")
    print(articles.iloc[selected_index]["headline"])

    recommendations = recommend_articles(
        selected_index,
        articles,
        tfidf_matrix,
        top_n=5
    )

    print("\nRecommended articles:")
    print(
        recommendations[
            ["category", "headline", "similarity_score"]
        ].to_string(index=False)
    )
