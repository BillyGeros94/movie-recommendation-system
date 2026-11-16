import random
from datetime import date

import streamlit as st
import joblib
import pandas as pd
import requests
from pathlib import Path

from recommender import MIN_RATINGS, recommend_uv, search_movies

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬"
)

# -------------------------
# Christmas theme: blinking lights along the top and gentle snowfall, on its own from
# 1 December to 6 January. Clicks pass through; reduced-motion settings are respected.
# -------------------------

LIGHT_COLOURS = ["#e53935", "#fdd835", "#43a047", "#1e88e5", "#fb8c00"]


def is_festive(today=None):
    """True from 1 December to 6 January."""
    today = today or date.today()
    return today.month == 12 or (today.month == 1 and today.day <= 6)


def _lights_html(n_bulbs=40):
    bulbs = "".join(
        f'<span class="xmas-bulb" style="background:{c};color:{c};animation-delay:{(i % 5) * 0.3:.1f}s"></span>'
        for i, c in enumerate(LIGHT_COLOURS[i % len(LIGHT_COLOURS)] for i in range(n_bulbs))
    )
    return f'<div class="xmas-lights">{bulbs}</div>'


def _snow_html(n_flakes=25):
    rng = random.Random(7)  # fixed seed: flakes keep their places when Streamlit reruns
    flakes = []
    for _ in range(n_flakes):
        left = rng.uniform(0, 100)
        size = rng.uniform(0.6, 1.3)
        duration = rng.uniform(9, 18)
        delay = -rng.uniform(0, 18)  # negative delay: snow is already falling on page load
        opacity = rng.uniform(0.4, 0.9)
        flakes.append(
            f'<span class="xmas-flake" style="left:{left:.1f}vw;font-size:{size:.2f}rem;'
            f'opacity:{opacity:.2f};animation-duration:{duration:.1f}s,{duration / 3:.1f}s;'
            f'animation-delay:{delay:.1f}s,{delay:.1f}s">❄</span>'
        )
    return f'<div class="xmas-snow">{"".join(flakes)}</div>'


_CSS = """
<style>
/* lights along the top edge, above Streamlit's header; clicks pass through */
.xmas-lights {
    position: fixed; top: 0; left: 0; width: 100%; height: 22px;
    display: flex; justify-content: space-around; align-items: flex-start;
    border-top: 2px solid #2e3b2e;
    pointer-events: none; z-index: 1000001;
}
.xmas-bulb {
    width: 9px; height: 13px; margin-top: 3px;
    border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
    box-shadow: 0 2px 10px 2px currentColor;
    animation: xmas-blink 1.5s ease-in-out infinite alternate;
}
@keyframes xmas-blink {
    from { opacity: 1; }
    to   { opacity: 0.25; box-shadow: none; }
}

/* falling snow over the whole page; clicks pass through */
.xmas-snow {
    position: fixed; inset: 0; overflow: hidden;
    pointer-events: none; z-index: 1000000;
}
.xmas-flake {
    position: absolute; top: -2rem;
    color: #b3d4f5; text-shadow: 0 0 3px rgba(255, 255, 255, 0.8);
    animation-name: xmas-fall, xmas-sway;
    animation-timing-function: linear, ease-in-out;
    animation-iteration-count: infinite, infinite;
    animation-direction: normal, alternate;
}
@keyframes xmas-fall {
    to { top: 105vh; }
}
@keyframes xmas-sway {
    from { transform: translateX(-15px); }
    to   { transform: translateX(15px); }
}

/* people who turned off animations in their system settings get none */
@media (prefers-reduced-motion: reduce) {
    .xmas-snow { display: none; }
    .xmas-bulb { animation: none; }
}
</style>
"""


def add_christmas_theme(force=False, snow=True, lights=True):
    """Add the theme if it is the festive season (or force=True)."""
    if not (force or is_festive()):
        return
    html = _CSS
    if lights:
        html += _lights_html()
    if snow:
        html += _snow_html()
    st.markdown(html, unsafe_allow_html=True)


add_christmas_theme()

# -------------------------
# Configuration
# -------------------------

TMDB_BASE_URL = "https://api.themoviedb.org/3/movie/"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w200"

# Posters are optional, so the app also runs without a TMDB key
try:
    TMDB_API_KEY = st.secrets.get("TMDB_API_KEY")
except Exception:
    TMDB_API_KEY = None

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

# Only movies the model knows can be searched and rated
movies_df = pd.DataFrame({"movieId": model["movie_ids"], "title": model["titles"]})
titles_by_id = dict(zip(movies_df["movieId"].tolist(), movies_df["title"].tolist()))

# -------------------------
# Poster fetching
# -------------------------

@st.cache_data
def get_poster(movie_id):
    if not TMDB_API_KEY:
        return None
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
# Streamlit UI
# -------------------------

st.title("🎬 Movie Recommendation System")
st.write(f"Rate at least {MIN_RATINGS} movies and get personalised recommendations.")

if "ratings" not in st.session_state:
    st.session_state.ratings = {}

# -------------------------
# Movie search
# -------------------------

search_query = st.text_input("Search for a movie")

if search_query:
    filtered = search_movies(movies_df, search_query)
    if not filtered.empty:
        # Select by movieId, because different movies can share a title
        movie_id = st.selectbox(
            "Select a movie",
            filtered["movieId"].tolist(),
            format_func=lambda m: titles_by_id[m]
        )

        selected_rating = st.slider(
            "Your rating",
            min_value=0.5,
            max_value=5.0,
            value=4.0,
            step=0.5
        )

        if st.button("Add movie"):
            st.session_state.ratings[movie_id] = selected_rating
            st.success(f"Added **{titles_by_id[movie_id]}** with rating {selected_rating:.1f}/5.0")
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
        poster_url = get_poster(movie_id)

        col1, col2, col3 = st.columns([1, 4, 1])

        with col1:
            if poster_url:
                st.image(poster_url, width=60)

        with col2:
            st.write(f"🎬 **{titles_by_id[movie_id]}** — {float(rating):.1f}/5.0")

        with col3:
            if st.button("❌", key=f"delete_{movie_id}"):
                del st.session_state.ratings[movie_id]
                st.rerun()

# -------------------------
# Recommend
# -------------------------

st.write(f"Movies rated: {len(st.session_state.ratings)}")

if len(st.session_state.ratings) >= MIN_RATINGS:
    if st.button("Get Recommendations"):
        results = recommend_uv(model, st.session_state.ratings, n=10)

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
    st.warning(f"Please rate at least {MIN_RATINGS} movies first.")

st.caption(
    "Ratings data: MovieLens, GroupLens Research. "
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)
