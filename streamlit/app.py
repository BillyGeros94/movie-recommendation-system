import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from pathlib import Path

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬"
)

# -------------------------
# Configuration
# -------------------------

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
TMDB_BASE_URL = "https://api.themoviedb.org/3/movie/"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w200"

# -------------------------
# Load model and data
# -------------------------

@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / "uv_model_deploy.joblib")

@st.cache_resource
def load_links():
    return pd.read_csv(Path(__file__).parent / "links.csv")

model = load_model()
links_df = load_links()

Q = model["Q"]
b_i = model["b_i"]
b_u_mean = model["b_u_mean"]
global_mean = model["global_mean"]
movie_idx = model["movie_idx"]
movies_df = model["movies_df"]

# -------------------------
# Poster fetching
# -------------------------

@st.cache_data
def get_poster(movie_id):
    row = links_df[links_df["movieId"] == movie_id]
    if row.empty or pd.isna(row["tmdbId"].iloc[0]):
        return None
    tmdb_id = int(row["tmdbId"].iloc[0])
    try:
        response = requests.get(
            f"{TMDB_BASE_URL}{tmdb_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=5
        )
        path = response.json().get("poster_path")
        return f"{TMDB_IMG_BASE}{path}" if path else None
    except Exception:
        return None

# -------------------------
# Recommendation function
# -------------------------

def recommend_uv(ratings_dict, n=10):

    rated_movie_indices = []
    ratings = []

    for movie_id, rating in ratings_dict.items():
        if movie_id in movie_idx:
            rated_movie_indices.append(movie_idx[movie_id])
            ratings.append(rating)

    rated_movie_indices = np.array(rated_movie_indices)
    ratings = np.array(ratings)

    if len(rated_movie_indices) < 5:
        return pd.DataFrame()

    weights = ratings / ratings.sum()
    P_new = np.average(Q[rated_movie_indices], axis=0, weights=weights)

    predicted_scores = global_mean + b_u_mean + b_i + np.dot(Q, P_new)

    recommendations = pd.DataFrame({
        "movieId": list(movie_idx.keys()),
        "predicted_rating": predicted_scores
    })

    recommendations = recommendations[
        ~recommendations["movieId"].isin(ratings_dict.keys())
    ]

    result = (
        recommendations
        .nlargest(n, "predicted_rating")
        .merge(movies_df, on="movieId")
    )

    result["predicted_rating"] = (
        result["predicted_rating"]
        .clip(0.5, 5.0)
        .round(2)
    )

    return result[["movieId", "title", "predicted_rating"]]

# -------------------------
# Streamlit UI
# -------------------------

st.title("🎬 Movie Recommendation System")
st.write("Rate at least 5 movies and get personalised recommendations.")

if "ratings" not in st.session_state:
    st.session_state.ratings = {}

# -------------------------
# Movie search
# -------------------------

search_query = st.text_input("Search for a movie")

if search_query:
    filtered = movies_df[
        movies_df["title"].str.contains(search_query, case=False, na=False)
    ]
    if not filtered.empty:
        selected_movie = st.selectbox("Select a movie", filtered["title"].tolist())

        selected_rating = st.slider(
            "Your rating",
            min_value=0.5,
            max_value=5.0,
            value=4.0,
            step=0.5
        )

        if st.button("Add movie"):
            movie_id = movies_df.loc[
                movies_df["title"] == selected_movie, "movieId"
            ].iloc[0]
            st.session_state.ratings[movie_id] = selected_rating
            st.success(f"Added **{selected_movie}** with rating {selected_rating:.1f}/5.0")
    else:
        st.warning("No movies found. Try a different search.")

# -------------------------
# Show current ratings
# -------------------------

st.subheader("Your ratings")

if len(st.session_state.ratings) == 0:
    st.info("No movies rated yet.")
else:
    for movie_id, rating in list(st.session_state.ratings.items()):
        title = movies_df.loc[
            movies_df["movieId"] == movie_id, "title"
        ].iloc[0]

        poster_url = get_poster(movie_id)

        col1, col2, col3 = st.columns([1, 4, 1])

        with col1:
            if poster_url:
                st.image(poster_url, width=60)

        with col2:
            st.write(f"🎬 **{title}** — {float(rating):.1f}/5.0")

        with col3:
            if st.button("❌", key=f"delete_{movie_id}"):
                del st.session_state.ratings[movie_id]
                st.rerun()

# -------------------------
# Recommend
# -------------------------

st.write(f"Movies rated: {len(st.session_state.ratings)}")

if len(st.session_state.ratings) >= 5:
    if st.button("Get Recommendations"):
        results = recommend_uv(st.session_state.ratings, n=10)

        if isinstance(results, pd.DataFrame) and not results.empty:
            st.subheader("Recommended movies")
            for _, row in results.iterrows():
                poster_url = get_poster(row["movieId"])
                col1, col2 = st.columns([1, 4])
                with col1:
                    if poster_url:
                        st.image(poster_url, width=60)
                with col2:
                    st.write(f"🎬 **{row['title']}**")
        else:
            st.error("Could not generate recommendations. Please try rating different movies.")
else:
    st.warning("Please rate at least 5 movies first.")
