import numpy as np
import pandas as pd
import pytest

from recommender import MIN_RATINGS, recommend_uv, search_movies
from uv_model import UVDecomposition

@pytest.fixture(scope='module')
def demo_model(ratings, movies):
  uv = UVDecomposition(n_factors=4, std=0.1)
  uv.fit(ratings, movies, n_epochs=150, lr=0.02, batch_size=500)
  uv.fit_ease(ratings)
  return {
    'Q': uv.Q, 'b_i': uv.b_i, 'global_mean': uv.global_mean, 'fold_in_reg': uv.fold_in_reg, 'candidates': uv.candidates,
    'ease_weights': uv.ease_weights, 'shortlist_size': uv.shortlist_size, 'liked_rating': uv.liked_rating,
    'movie_ids': uv.movie_ids, 'titles': np.array([uv.titles[int(m)] for m in uv.movie_ids], dtype=object)
  }

@pytest.mark.parametrize('query', ['(', '(1995', '*batteries', '[', 'Toy Story ('])
def test_search_treats_symbols_as_text(query):
  movies = pd.DataFrame({'movieId': [1], 'title': ['Toy Story (1995)']})
  search_movies(movies, query)

def test_search_matches_brackets_literally():
  movies = pd.DataFrame({'movieId': [1, 2], 'title': ['Toy Story (1995)', 'Heat (1995']})
  assert search_movies(movies, 'story (1995').movieId.tolist() == [1]

def test_likes_and_dislikes_give_different_recommendations(demo_model):
  even_movies = [2, 4, 6, 8, 10]
  liked = recommend_uv(demo_model, {m: 5.0 for m in even_movies})
  disliked = recommend_uv(demo_model, {m: 0.5 for m in even_movies})
  assert not set(liked['movieId']) & set(disliked['movieId'])

def test_rated_movies_are_not_recommended(demo_model):
  ratings = {1: 4.0, 2: 3.0, 3: 5.0, 4: 2.0, 5: 4.5}
  assert not set(ratings) & set(recommend_uv(demo_model, ratings)['movieId'])

def test_too_few_ratings_give_nothing(demo_model):
  assert recommend_uv(demo_model, {m: 4.0 for m in range(1, MIN_RATINGS)}).empty

def test_only_candidates_are_recommended(demo_model):
  allowed = demo_model['candidates'][:20]
  model = {**demo_model, 'candidates': allowed, 'ease_weights': None}
  result = recommend_uv(model, {m: 4.0 for m in range(1, 6)})
  assert set(result['movieId']) <= set(demo_model['movie_ids'][allowed].tolist())

def test_recommendations_come_from_the_shortlist(demo_model):
  model = {**demo_model, 'shortlist_size': 12}
  ratings = {m: 5.0 for m in [2, 4, 6, 8, 10]}
  result = recommend_uv(model, ratings, n=10)
  assert len(result) == 10
  assert not set(ratings) & set(result['movieId'])

def test_all_low_ratings_still_get_recommendations(demo_model):
  assert len(recommend_uv(demo_model, {m: 0.5 for m in [2, 4, 6, 8, 10]})) == 10

def test_numpy_movie_ids_are_accepted(demo_model):
  ratings = {np.int64(m): 4.0 for m in range(1, 6)}
  assert len(recommend_uv(demo_model, ratings)) == 10
