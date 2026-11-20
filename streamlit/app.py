"""Movie Recommender: rate a few films and get recommendations from a matrix factorisation model.

Run locally from the project folder:  streamlit run streamlit/app.py
"""
import random
import re
from datetime import date

import streamlit as st
import joblib
import pandas as pd
import requests
from pathlib import Path

import ui
from recommender import MIN_RATINGS, recommend_uv, search_movies

st.set_page_config(
    page_title="Movie Recommender · Rate a few films, find the next one",
    page_icon=":material/movie:"
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
# Interface
# -------------------------

BRAND = "Movie Recommender"
GLYPH = '<path class="f" d="M10.5 8.2 L18.6 13 L10.5 17.8 Z"/>'   # play
EXTRA_CSS = """
.poster { width: 100%; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 8px; display: block;
  background: var(--soft); border: 1px solid var(--line); }
.tile { width: 100%; aspect-ratio: 2 / 3; border-radius: 8px; background: var(--soft); border: 1px solid var(--line);
  display: flex; align-items: flex-end; padding: .8rem; box-sizing: border-box; font-size: .98rem; font-weight: 600;
  letter-spacing: -.02em; line-height: 1.2; color: var(--ink); font-family: 'Inter', system-ui, sans-serif; }
.thumb { width: 34px; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 4px; background: var(--soft); display: block; }
.rec { margin-bottom: 1.2rem; }
.rec .mono { margin: .55rem 0 .15rem; }
.rec .t { font-size: .88rem; line-height: 1.3; color: var(--ink); font-family: 'Inter', system-ui, sans-serif; }
.rated { display: flex; align-items: center; gap: .8rem; min-height: 2.6rem; }
.rated .t { font-size: .95rem; color: var(--ink); font-family: 'Inter', system-ui, sans-serif; }
[class*="st-key-rated_"] { border-top: 1px solid var(--line); padding: .35rem 0; }
[class*="st-key-rated_"] button { border: 0 !important; background: transparent; color: var(--muted); }
[class*="st-key-rated_"] .stButton { display: flex; justify-content: flex-end; }
[class*="st-key-rated_"] button:hover { color: var(--ink); }
.st-key-progress .mono { text-align: right; }
"""


def nice(title):
    """MovieLens titles as people write them: "Matrix, The (1999)" -> "The Matrix (1999)", without the
    alternative titles in brackets."""
    m = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", title)
    if not m:
        return title
    name, year = m.groups()
    name = re.sub(r"\s*\([^)]*\)", "", name).strip()
    article = re.match(r"^(.*), (The|A|An)$", name)
    if article:
        name = f"{article[2]} {article[1]}"
    return f"{name} ({year})"


def poster_html(movie_id, title, cls="poster"):
    """The poster, or a quiet title tile when there is none (no TMDB key, or no poster)."""
    url = get_poster(movie_id)
    if url:
        return f'<img class="{cls}" src="{url}" alt="{title}">', True
    if cls == "thumb":
        return '<div class="thumb"></div>', False
    name, _, year = title.rpartition(" (")
    return f'<div class="tile"><div>{name}<div class="mono" style="margin-top:.35rem">{year.rstrip(")")}</div></div></div>', False


st.session_state.setdefault("ratings", {})
ratings = st.session_state.ratings
ui.apply(width="900px", extra=EXTRA_CSS)

right = ui.top_bar(BRAND, GLYPH, widths=(5, 1.6))
with right:
    st.markdown(f'<div class="mono" style="text-align:right">{min(len(ratings), MIN_RATINGS)} / {MIN_RATINGS} rated</div>'
                if len(ratings) < MIN_RATINGS else
                f'<div class="mono" style="text-align:right">{len(ratings)} rated</div>', unsafe_allow_html=True)

ui.hero(f"{len(movies_df):,} films · MovieLens ratings · Matrix factorisation",
        "Rate a few films,", "find the next one.",
        f"Rate at least {MIN_RATINGS} films you have seen. The model places your taste among other viewers' ratings "
        "and recommends the films you are most likely to rate highly.")

# ---- 01 Rate ----
ui.label("01 · Rate a film")
search_query = st.text_input("Search", placeholder="Search by title", label_visibility="collapsed")
if search_query:
    filtered = search_movies(movies_df, search_query)
    if not filtered.empty:
        left, mid, add = st.columns([3.2, 2, 1.1], vertical_alignment="bottom")
        # Select by movieId, because different movies can share a title
        movie_id = left.selectbox("Film", filtered["movieId"].tolist(), format_func=lambda m: nice(titles_by_id[m]))
        selected_rating = mid.slider("Your rating", min_value=0.5, max_value=5.0, value=4.0, step=0.5, format="%.1f")
        if add.button("Add", type="primary", width="stretch"):
            ratings[movie_id] = selected_rating
            st.session_state.pop("recs", None)
            st.toast(f"Rated {nice(titles_by_id[movie_id])}: {selected_rating:.1f} / 5")
            st.rerun()
    else:
        st.markdown('<div class="note">No films match that title. Try fewer words or another spelling.</div>',
                    unsafe_allow_html=True)

# ---- 02 Your ratings ----
ui.label(f"02 · Your ratings · {len(ratings)}")
if not ratings:
    st.markdown('<div class="note">Films you rate appear here.</div>', unsafe_allow_html=True)
for movie_id, rating in list(ratings.items()):
    with st.container(key=f"rated_{movie_id}"):
        a, b, c = st.columns([8, 1.4, 1.1], vertical_alignment="center")
        title = nice(titles_by_id[movie_id])
        a.markdown(f'<div class="rated">{poster_html(movie_id, title, "thumb")[0]}'
                   f'<span class="t">{title}</span></div>', unsafe_allow_html=True)
        b.markdown(f'<div class="mono">{float(rating):.1f} / 5</div>', unsafe_allow_html=True)
        if c.button("Remove", key=f"delete_{movie_id}", type="tertiary"):
            del ratings[movie_id]
            st.session_state.pop("recs", None)
            st.rerun()

# ---- 03 Recommendations ----
ui.label("03 · Recommendations")
missing = MIN_RATINGS - len(ratings)
if missing > 0:
    st.markdown(f'<div class="note">Rate {missing} more film{"s" if missing > 1 else ""} to see your '
                'recommendations.</div>', unsafe_allow_html=True)
elif "recs" not in st.session_state:
    if st.button("Get recommendations", type="primary"):
        st.session_state.recs = recommend_uv(model, ratings, n=10)
        st.rerun()
else:
    results = list(st.session_state.recs.itertuples())
    for start in range(0, len(results), 5):          # one row of five at a time, so the rows line up
        cols = st.columns(5)
        for i, row in enumerate(results[start:start + 5], start=start):
            title = nice(row.title)
            art, has_poster = poster_html(row.movieId, title)
            cols[i - start].markdown(f'<div class="rec">{art}<div class="mono">{i + 1:02d} · '
                                     f'{row.predicted_rating:.1f} predicted</div>'
                                     + (f'<div class="t">{title}</div>' if has_poster else "") + '</div>',
                                     unsafe_allow_html=True)
    st.markdown('<div class="note">Predicted: the rating the model expects you to give, out of 5. '
                'Recommendations update when your ratings change.</div>', unsafe_allow_html=True)

ui.foot("Ratings: MovieLens, GroupLens Research · Posters: TMDB. This product uses the TMDB API but is not "
        "endorsed or certified by TMDB.")
