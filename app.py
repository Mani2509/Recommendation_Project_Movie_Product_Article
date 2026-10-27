
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from services.movie_recommender import (
    recommend_movies,
    evaluate_movie_model,
)

from services.article_recommender import (
    load_articles,
    build_article_model,
    recommend_articles,
    evaluate_article_model,
)

from services.product_recommender import (
    load_products,
    build_product_model,
    recommend_products,
    get_product_image,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real-Time Recommendation Service",
    page_icon="🎯",
    layout="wide",
)


# ============================================================
# MOVIE CONFIGURATION
# ============================================================

# Prefer setting TMDB_API_KEY as an environment variable.
# Do not publish API keys in public repositories.
TMDB_API_KEY = "7b3104e3dfa1db9ee30627575a369a99"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"


def clean_movie_title(title):
    title = str(title).strip()

    for article in ["The", "A", "An"]:
        suffix = f", {article}"

        if title.lower().endswith(suffix.lower()):
            title = article + " " + title[:-len(suffix)]
            break

    return title


@st.cache_data(show_spinner=False)
def get_movie_poster(movie_title):
    title = clean_movie_title(movie_title)

    year_match = re.search(r"\((\d{4})\)", title)
    year = year_match.group(1) if year_match else None

    search_title = re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        title,
    ).strip()

    if not TMDB_API_KEY:
        return None

    params = {
        "api_key": TMDB_API_KEY,
        "query": search_title,
    }

    if year:
        params["year"] = year

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=10,
        )
        response.raise_for_status()

        movies = response.json().get("results", [])

        for movie in movies:
            poster_path = movie.get("poster_path")

            if poster_path:
                return f"{TMDB_IMAGE_URL}{poster_path}"

    except requests.RequestException as error:
        print(f"TMDB poster search failed for {title}: {error}")

    return None


# ============================================================
# ARTICLE RECOMMENDATION MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def get_article_recommendation_model():
    articles = load_articles()
    vectorizer, tfidf_matrix = build_article_model(articles)

    return articles, vectorizer, tfidf_matrix


# ============================================================
# PRODUCT RECOMMENDATION MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def get_product_recommendation_model():
    products = load_products()
    products, vectorizer, product_matrix = build_product_model(products)

    return products, vectorizer, product_matrix


# ============================================================
# PRODUCT MODEL EVALUATION
# ============================================================

def evaluate_product_model(
    products,
    product_matrix,
    sample_size=200,
    top_n=5,
    random_state=42,
):
    """
    Proxy evaluation:
    Products with the same subCategory as the selected product
    are treated as relevant.

    These are category-based proxy metrics, not human-judged scores.
    """

    if "subCategory" not in products.columns:
        raise ValueError("Product data must contain subCategory.")

    if len(products) < 2:
        raise ValueError("At least two products are required.")

    valid_indices = np.arange(len(products))
    rng = np.random.default_rng(random_state)

    sample_size = min(sample_size, len(valid_indices))
    sampled_indices = rng.choice(
        valid_indices,
        size=sample_size,
        replace=False,
    )

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    categories = products["subCategory"].fillna("").astype(str).to_numpy()

    for product_index in sampled_indices:
        selected_category = categories[product_index]

        if not selected_category:
            continue

        relevant_mask = categories == selected_category
        relevant_mask[product_index] = False

        total_relevant = int(relevant_mask.sum())

        if total_relevant == 0:
            continue

        similarity_scores = (
            product_matrix[product_index] @ product_matrix.T
        ).toarray().ravel()

        # Never recommend the selected product itself.
        similarity_scores[product_index] = -np.inf

        top_indices = np.argsort(similarity_scores)[-top_n:][::-1]

        relevance = relevant_mask[top_indices].astype(int)
        relevant_recommended = int(relevance.sum())

        precision_scores.append(relevant_recommended / top_n)
        recall_scores.append(relevant_recommended / total_relevant)

        discounts = np.log2(np.arange(2, len(relevance) + 2))
        dcg = float(np.sum(relevance / discounts))

        ideal_count = min(top_n, total_relevant)
        ideal_relevance = np.ones(ideal_count, dtype=float)
        ideal_discounts = np.log2(
            np.arange(2, ideal_count + 2)
        )
        idcg = float(np.sum(ideal_relevance / ideal_discounts))

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    if not precision_scores:
        raise ValueError(
            "Could not evaluate products using subCategory labels."
        )

    return {
        "Precision@5": float(np.mean(precision_scores)),
        "Recall@5": float(np.mean(recall_scores)),
        "NDCG@5": float(np.mean(ndcg_scores)),
        "Evaluated Products": len(precision_scores),
    }


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "recommendation_type" not in st.session_state:
    st.session_state.recommendation_type = "Movie"

if "user_id" not in st.session_state:
    st.session_state.user_id = 101


USER_IDS = [101, 102, 103, 104, 105]


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    st.title("🎯 Real-Time Recommendation Service")
    st.caption(
        "Personalized recommendations using Machine Learning"
    )
    st.divider()

    st.subheader("Choose Your Recommendation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎬 Movie Recommendation")
        st.write("Discover movies you'll love.")

        if st.button("Select Movies", use_container_width=True):
            st.session_state.recommendation_type = "Movie"
            st.session_state.page = "selection"
            st.rerun()

    with col2:
        st.markdown("### 🛍️ Product Recommendation")
        st.write("Discover similar fashion products.")

        if st.button("Select Products", use_container_width=True):
            st.session_state.recommendation_type = "Product"
            st.session_state.page = "selection"
            st.rerun()

    with col3:
        st.markdown("### 📰 Article Recommendation")
        st.write("Discover relevant articles.")

        if st.button("Select Articles", use_container_width=True):
            st.session_state.recommendation_type = "Article"
            st.session_state.page = "selection"
            st.rerun()


# ============================================================
# SELECTION PAGE
# ============================================================

elif st.session_state.page == "selection":

    st.title("⚙️ Recommendation Settings")

    st.write(
        f"Selected Type: {st.session_state.recommendation_type}"
    )
    st.divider()

    recommendation_options = ["Movie", "Product", "Article"]

    selected_type = st.selectbox(
        "Select Recommendation Type",
        recommendation_options,
        index=recommendation_options.index(
            st.session_state.recommendation_type
        ),
        key="recommendation_type_selector",
    )

    if selected_type == "Movie":

        selected_user = st.selectbox(
            "Select User ID",
            USER_IDS,
            index=(
                USER_IDS.index(st.session_state.user_id)
                if st.session_state.user_id in USER_IDS
                else 0
            ),
            key="user_id_selector",
        )

    else:
        selected_user = None

        if selected_type == "Article":
            st.info(
                "Select an article on the next page to get similar "
                "articles. User ID is not required."
            )

        elif selected_type == "Product":
            st.info(
                "Search for a fashion product and select it to "
                "find similar products. User ID is not required."
            )

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Get Recommendations",
            use_container_width=True,
        ):
            st.session_state.recommendation_type = selected_type

            if selected_type == "Movie":
                st.session_state.user_id = selected_user

            st.session_state.page = "results"
            st.rerun()

    with col2:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()


# ============================================================
# RESULTS PAGE
# ============================================================

elif st.session_state.page == "results":

    recommendation_type = st.session_state.recommendation_type

    st.title("🎯 Recommendation Results")
    st.write(f"**Recommendation Type:** {recommendation_type}")

    if recommendation_type == "Movie":
        user_id = st.session_state.user_id
        st.write(f"**User ID:** {user_id}")

    st.divider()

    # ========================================================
    # MOVIE RECOMMENDATION
    # ========================================================

    if recommendation_type == "Movie":

        st.subheader("🎬 Recommended Movies")

        try:
            with st.spinner("Generating movie recommendations..."):
                results = recommend_movies(user_id, top_n=10)

            if results.empty:
                st.warning(
                    "No recommendations available for this user."
                )

            else:
                st.success(
                    f"Found {len(results)} movie recommendations!"
                )

                for _, movie in results.iterrows():

                    original_title = movie["title"]
                    title = clean_movie_title(original_title)
                    poster_url = get_movie_poster(title)

                    with st.container(border=True):

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            if poster_url:
                                st.image(poster_url, width=160)
                            else:
                                st.write("🎬 Poster not available")

                        with col2:
                            st.subheader(title)

                            st.metric(
                                "Predicted Rating",
                                f"{movie['predicted_rating']:.2f} / 5",
                            )

        except FileNotFoundError:
            st.error(
                "Movie dataset not found. Check "
                "data/movies/u.data and data/movies/u.item."
            )

        except Exception as error:
            st.error(
                f"Could not generate movie recommendations: {error}"
            )

        # Movie evaluation metrics
        st.divider()
        st.header("📊 Movie Model Evaluation Metrics")

        try:
            with st.spinner("Evaluating movie recommendation model..."):
                metrics = evaluate_movie_model()

            if metrics["RMSE"] is not None:

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("RMSE", f"{metrics['RMSE']:.4f}")
                col2.metric("MAE", f"{metrics['MAE']:.4f}")
                col3.metric(
                    "Precision",
                    f"{metrics['Precision']:.2%}",
                )
                col4.metric(
                    "Recall",
                    f"{metrics['Recall']:.2%}",
                )

                st.caption(
                    "RMSE and MAE measure rating prediction errors. "
                    "Precision and Recall use a rating threshold of 4/5."
                )

                metric_names = [
                    "RMSE",
                    "MAE",
                    "Precision",
                    "Recall",
                ]

                metric_values = [
                    metrics["RMSE"],
                    metrics["MAE"],
                    metrics["Precision"],
                    metrics["Recall"],
                ]

                chart_data = pd.DataFrame({
                    "Metric": metric_names,
                    "Score": metric_values,
                })

                st.subheader("📈 Movie Evaluation Chart")
                st.bar_chart(chart_data, x="Metric", y="Score")

            else:
                st.warning(
                    "Unable to calculate movie evaluation metrics."
                )

        except Exception as error:
            st.error(f"Movie evaluation failed: {error}")

    # ========================================================
    # PRODUCT RECOMMENDATION
    # ========================================================

    elif recommendation_type == "Product":

        st.subheader("🛍️ Fashion Product Recommendations")

        try:
            with st.spinner(
                "Loading product dataset and building model..."
            ):
                products, vectorizer, product_matrix = (
                    get_product_recommendation_model()
                )

            st.success(
                f"Product model ready! {len(products):,} products loaded."
            )

            st.caption(
                "Search by product name, then select a product "
                "to find similar products."
            )

            # Product search
            product_search = st.text_input(
                "🔎 Search for a product",
                placeholder="Example: Navy Blue Shirt, Jeans, Watch",
                key="product_search",
            )

            if product_search.strip():
                matched_products = products[
                    products["productDisplayName"].str.contains(
                        product_search.strip(),
                        case=False,
                        na=False,
                        regex=False,
                    )
                ].head(200)
            else:
                matched_products = products.head(100)

            if matched_products.empty:
                st.warning(
                    "No matching products found. Try another keyword."
                )

            else:
                selected_product_id = st.selectbox(
                    "Choose a product",
                    matched_products["id"].tolist(),
                    format_func=lambda product_id: (
                        f"{product_id} — "
                        f"{products.loc[products['id'] == product_id, 'productDisplayName'].iloc[0]}"
                    ),
                    key="selected_product_id",
                )

                selected_rows = products[
                    products["id"] == selected_product_id
                ]

                if not selected_rows.empty:
                    selected_product = selected_rows.iloc[0]

                    st.subheader("Selected Product")

                    image_path = get_product_image(selected_product_id)

                    # Fallback image directory if necessary.
                    if not image_path:
                        fallback_path = (
                            Path(__file__).resolve().parent
                            / "data"
                            / "products"
                            / "myntradataset"
                            / "images"
                            / f"{selected_product_id}.jpg"
                        )

                        if fallback_path.exists():
                            image_path = str(fallback_path)

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        if image_path:
                            st.image(image_path, width=240)
                        else:
                            st.info("Product image not available.")

                    with col2:
                        st.markdown(
                            f"### {selected_product['productDisplayName']}"
                        )

                        st.write(
                            f"**Product ID:** {selected_product['id']}"
                        )
                        st.write(
                            f"**Gender:** {selected_product['gender']}"
                        )
                        st.write(
                            f"**Category:** {selected_product['masterCategory']}"
                        )
                        st.write(
                            f"**Subcategory:** {selected_product['subCategory']}"
                        )
                        st.write(
                            f"**Type:** {selected_product['articleType']}"
                        )
                        st.write(
                            f"**Colour:** {selected_product['baseColour']}"
                        )

                    if st.button(
                        "Recommend Similar Products",
                        use_container_width=True,
                    ):

                        with st.spinner(
                            "Finding similar products..."
                        ):
                            product_results = recommend_products(
                                int(selected_product_id),
                                products,
                                product_matrix,
                                top_n=5,
                            )

                        st.subheader("✨ Recommended Similar Products")

                        if product_results.empty:
                            st.warning("No similar products found.")

                        else:
                            cols = st.columns(5)

                            for position, (_, product) in enumerate(
                                product_results.iterrows()
                            ):
                                with cols[position % 5]:
                                    with st.container(border=True):

                                        rec_image = get_product_image(
                                            product["id"]
                                        )

                                        if not rec_image:
                                            rec_fallback = (
                                                Path(__file__).resolve().parent
                                                / "data"
                                                / "products"
                                                / "myntradataset"
                                                / "images"
                                                / f"{int(product['id'])}.jpg"
                                            )

                                            if rec_fallback.exists():
                                                rec_image = str(rec_fallback)

                                        if rec_image:
                                            st.image(
                                                rec_image,
                                                use_container_width=True,
                                            )
                                        else:
                                            st.caption(
                                                "Image not available"
                                            )

                                        st.markdown(
                                            f"**{product['productDisplayName']}**"
                                        )

                                        st.caption(
                                            f"Category: {product['subCategory']}"
                                        )
                                        st.caption(
                                            f"Colour: {product['baseColour']}"
                                        )

                                        st.metric(
                                            "Similarity",
                                            f"{product['similarity_score']:.2%}",
                                        )

            # Product model evaluation metrics
            st.divider()
            st.header("📊 Product Model Evaluation Metrics")

            st.caption(
                "Proxy evaluation: products in the same subcategory "
                "are treated as relevant. These scores are estimates, "
                "not human-judged relevance."
            )

            try:
                with st.spinner(
                    "Evaluating product recommendation model..."
                ):
                    product_metrics = evaluate_product_model(
                        products,
                        product_matrix,
                        sample_size=200,
                        top_n=5,
                    )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Precision@5",
                    f"{product_metrics['Precision@5']:.2%}",
                )
                col2.metric(
                    "Recall@5",
                    f"{product_metrics['Recall@5']:.2%}",
                )
                col3.metric(
                    "NDCG@5",
                    f"{product_metrics['NDCG@5']:.2%}",
                )

                st.caption(
                    f"Evaluated products: "
                    f"{product_metrics['Evaluated Products']}"
                )

                product_chart_data = pd.DataFrame({
                    "Metric": [
                        "Precision@5",
                        "Recall@5",
                        "NDCG@5",
                    ],
                    "Score": [
                        product_metrics["Precision@5"],
                        product_metrics["Recall@5"],
                        product_metrics["NDCG@5"],
                    ],
                })

                st.subheader("📈 Product Evaluation Chart")
                st.bar_chart(
                    product_chart_data,
                    x="Metric",
                    y="Score",
                )

            except Exception as error:
                st.error(f"Product evaluation failed: {error}")

        except FileNotFoundError as error:
            st.error(f"Product dataset not found: {error}")

        except Exception as error:
            st.error(
                f"Could not generate product recommendations: {error}"
            )

    # ========================================================
    # ARTICLE RECOMMENDATION
    # ========================================================

    elif recommendation_type == "Article":

        st.subheader("📰 Article Recommendations")

        try:
            with st.spinner(
                "Loading article dataset and building recommendation model..."
            ):
                articles, vectorizer, tfidf_matrix = (
                    get_article_recommendation_model()
                )

            st.success(
                f"Article model ready! {len(articles):,} articles loaded."
            )

            # Article evaluation metrics
            st.divider()
            st.header("📊 Article Model Evaluation Metrics")

            st.caption(
                "Proxy evaluation: articles in the same category "
                "are treated as relevant."
            )

            try:
                with st.spinner(
                    "Evaluating article recommendation model..."
                ):
                    article_metrics = evaluate_article_model(
                        articles,
                        tfidf_matrix,
                        sample_size=200,
                        top_n=5,
                    )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Precision@5",
                    f"{article_metrics['Precision@5']:.2%}",
                )
                col2.metric(
                    "Recall@5",
                    f"{article_metrics['Recall@5']:.2%}",
                )
                col3.metric(
                    "NDCG@5",
                    f"{article_metrics['NDCG@5']:.2%}",
                )

                st.caption(
                    f"Evaluated articles: "
                    f"{article_metrics['Evaluated Articles']}"
                )

                article_chart_data = pd.DataFrame({
                    "Metric": [
                        "Precision@5",
                        "Recall@5",
                        "NDCG@5",
                    ],
                    "Score": [
                        article_metrics["Precision@5"],
                        article_metrics["Recall@5"],
                        article_metrics["NDCG@5"],
                    ],
                })

                st.subheader("📈 Article Evaluation Chart")
                st.bar_chart(
                    article_chart_data,
                    x="Metric",
                    y="Score",
                )

            except Exception as error:
                st.error(f"Article evaluation failed: {error}")

            # Article search
            search_text = st.text_input(
                "🔎 Search for an article",
                placeholder="Enter a headline keyword, e.g. technology",
                key="article_search",
            )

            if search_text.strip():
                matched = articles[
                    articles["headline"].str.contains(
                        search_text.strip(),
                        case=False,
                        na=False,
                        regex=False,
                    )
                ].head(100)
            else:
                matched = articles.head(100)

            if matched.empty:
                st.warning(
                    "No matching articles found. Try another keyword."
                )

            else:
                selected_index = st.selectbox(
                    "Choose an article",
                    matched.index.tolist(),
                    format_func=lambda index: (
                        f"{articles.loc[index, 'category']} — "
                        f"{articles.loc[index, 'headline']}"
                    ),
                    key="selected_article_index",
                )

                with st.expander("Preview selected article"):
                    st.write(
                        articles.loc[
                            selected_index,
                            "short_description",
                        ]
                    )

                if st.button(
                    "Recommend Similar Articles",
                    use_container_width=True,
                ):

                    with st.spinner("Finding similar articles..."):
                        article_results = recommend_articles(
                            selected_index,
                            articles,
                            tfidf_matrix,
                            top_n=5,
                        )

                    st.subheader("📚 Recommended Similar Articles")

                    if article_results.empty:
                        st.warning("No similar articles found.")

                    else:
                        for _, row in article_results.iterrows():

                            with st.container(border=True):

                                st.caption(str(row["category"]))
                                st.subheader(str(row["headline"]))
                                st.write(str(row["short_description"]))

                                st.progress(
                                    max(
                                        0.0,
                                        min(
                                            1.0,
                                            float(row["similarity_score"]),
                                        ),
                                    )
                                )

                                st.caption(
                                    "Similarity Score: "
                                    f"{row['similarity_score']:.2%}"
                                )

                                link = row["link"]

                                if (
                                    pd.notna(link)
                                    and str(link).strip()
                                ):
                                    st.markdown(
                                        f"[Read article]({link})"
                                    )

        except FileNotFoundError as error:
            st.error(f"Article dataset not found: {error}")

        except Exception as error:
            st.error(
                f"Could not generate article recommendations: {error}"
            )


# ============================================================
# NAVIGATION
# ============================================================

if st.session_state.page == "results":

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "← Back to Selection",
            use_container_width=True,
        ):
            st.session_state.page = "selection"
            st.rerun()

    with col2:
        if st.button(
            "🏠 Back to Home",
            use_container_width=True,
        ):
            st.session_state.page = "home"
            st.rerun()
