import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬"
)

# -------------------------
# Load model
# -------------------------

@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / "uv_model_deploy.joblib")

model = load_model()

Q = model["Q"]
movie_idx = model["movie_idx"]
movies_df = model["movies_df"]

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

    predicted_scores = np.dot(Q, P_new)

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

    return result[["title", "predicted_rating"]]

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
            st.success(f"Added **{selected_movie}** with rating {selected_rating}/5")
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

        col1, col2 = st.columns([5, 1])

        with col1:
            st.write(f"🎬 **{title}** — {rating}/5")

        with col2:
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
                st.write(f"🎬 **{row['title']}** — {row['predicted_rating']}/5")
        else:
            st.error("Could not generate recommendations. Please try rating different movies.")
else:
    st.warning("Please rate at least 5 movies first.")