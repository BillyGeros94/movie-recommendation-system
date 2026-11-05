from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from fold_in import fold_in_user, score_new_user

MIN_RATINGS = 5

def search_movies(movies_df: pd.DataFrame, query: str) -> pd.DataFrame:

    # Plain-text match, so brackets and other symbols in a query are safe
    return movies_df[movies_df['title'].str.contains(query, case=False, na=False, regex=False)]

def recommend_uv(model: dict, ratings: dict, n: int = 10) -> pd.DataFrame:

    # Ratings are keyed by movieId. Every searchable movie is in the model
    movie_ids = np.asarray(model['movie_ids'])
    positions = {int(m): i for i, m in enumerate(movie_ids)}
    rated = {int(m): r for m, r in ratings.items() if int(m) in positions}

    if len(rated) < MIN_RATINGS:
        return pd.DataFrame(columns=['movieId', 'title', 'predicted_rating'])

    # Estimate the new user from their ratings, then score every movie
    item_indices = np.array([positions[m] for m in rated])
    b_u, p_u = fold_in_user(model['Q'], model['b_i'], model['global_mean'], item_indices,
                            np.array(list(rated.values())), reg=model['fold_in_reg'])
    scores = score_new_user(model['Q'], model['b_i'], model['global_mean'], b_u, p_u)

    # Keep the best unrated movies
    scores[item_indices] = -np.inf
    top = np.argsort(-scores, kind='stable')[:n]

    return pd.DataFrame({
        'movieId': movie_ids[top],
        'title': np.asarray(model['titles'])[top],
        'predicted_rating': np.clip(scores[top], 0.5, 5.0).round(2)
    })
