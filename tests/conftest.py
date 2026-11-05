import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'api'), str(ROOT / 'streamlit')]

@pytest.fixture(scope='session')
def movies():
  genres = ['Action|Thriller', 'Comedy', 'Horror|Thriller', 'Romance|Drama']
  return pd.DataFrame({
    'movieId': range(1, 61),
    'title': [f'Movie {i} (2000)' for i in range(1, 61)],
    'genres': [genres[i % 4] for i in range(60)]
  })

@pytest.fixture(scope='session')
def ratings():

  # Two taste groups: each user likes one half of the catalogue and dislikes the other
  rng = np.random.default_rng(0)
  rows = []
  for user_id in range(1, 81):
    group = user_id % 2
    for movie_id in rng.choice(np.arange(1, 61), 35, replace=False):
      liked = (movie_id % 2) == group
      rating = np.clip((4.5 if liked else 1.5) + rng.normal(0, 0.3), 0.5, 5.0)
      rows.append((user_id, int(movie_id), round(float(rating) * 2) / 2))
  return pd.DataFrame(rows, columns=['userId', 'movieId', 'rating'])
