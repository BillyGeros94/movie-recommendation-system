from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from fold_in import fold_in_user, rank_new_user, score_new_user

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

    # One calculation chooses the movies, the other estimates the rating shown with them
    item_indices = np.array([positions[m] for m in rated])
    values = np.array(list(rated.values()))
    ranking = rank_new_user(model['Q'], model['b_i'], model['global_mean'], item_indices, values,
                            model['candidates'], reg=model['fold_in_reg'])
    top = np.argsort(-ranking, kind='stable')[:n]
    top = top[np.isfinite(ranking[top])]

    b_u, p_u = fold_in_user(model['Q'], model['b_i'], model['global_mean'], item_indices, values, reg=model['fold_in_reg'])
    scores = score_new_user(model['Q'], model['b_i'], model['global_mean'], b_u, p_u)

    return pd.DataFrame({
        'movieId': movie_ids[top],
        'title': np.asarray(model['titles'])[top],
        'predicted_rating': np.clip(scores[top], 0.5, 5.0).round(2)
    })
