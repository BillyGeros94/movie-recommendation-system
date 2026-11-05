import numpy as np
import pytest

from baselines import BiasBaseline, ItemMeanBaseline, PopularityBaseline
from content_model import ContentBasedRecommender, split_genres
from fold_in import fold_in_user, score_new_user
from item_model import ItemBasedCF
from user_model import UserBasedCF
from uv_model import UVDecomposition

def fit_uv(ratings, movies):
  model = UVDecomposition(n_factors=4, std=0.1)
  model.fit(ratings, movies, n_epochs=150, lr=0.02, batch_size=500)
  return model

@pytest.fixture(scope='module')
def uv(ratings, movies):
  return fit_uv(ratings, movies)

@pytest.fixture(scope='module')
def item(ratings, movies):
  model = ItemBasedCF(n_neighbors=10, sim_shrinkage=5.0, block_size=16)
  model.fit(ratings, movies)
  return model

@pytest.fixture(scope='module')
def user(ratings, movies):
  model = UserBasedCF(n_neighbors=10, sim_shrinkage=5.0)
  model.fit(ratings, movies)
  return model

@pytest.fixture(scope='module')
def content(ratings, movies):
  model = ContentBasedRecommender()
  model.fit(ratings, movies)
  return model

def group_share(recommendations, group):
  return (recommendations['movieId'] % 2 == group).mean()

def test_uv_factors_start_centred_on_zero(ratings, movies):
  model = UVDecomposition(n_factors=8, std=0.05)
  model.fit(ratings, movies, n_epochs=1, lr=0.0)
  assert abs(model.Q.mean()) < 0.01 and abs(model.P.mean()) < 0.01

def test_uv_learns_the_taste_groups(uv, ratings):
  rmse = np.sqrt(np.mean((ratings['rating'].values - uv.predict(ratings['userId'].values, ratings['movieId'].values)) ** 2))
  assert rmse < 0.6

def test_uv_early_stopping_restores_the_best_epoch(ratings, movies):
  fit_part, val = ratings.iloc[::2], ratings.iloc[1::2]
  val = val[val['movieId'].isin(fit_part['movieId'])]
  model = UVDecomposition(n_factors=4, std=0.1)
  model.fit(fit_part, movies, n_epochs=200, lr=0.02, batch_size=500, val_df=val, patience=2)
  assert model._rmse(val) == pytest.approx(min(h['val_rmse'] for h in model.history), abs=1e-3)
  assert len(model.history) <= 200

def test_fold_in_separates_likes_from_dislikes(uv):
  even_movies = [2, 4, 6, 8, 10]
  liked = uv.recommend_new_user({m: 5.0 for m in even_movies}, n=10)
  disliked = uv.recommend_new_user({m: 0.5 for m in even_movies}, n=10)
  assert group_share(liked, 0) > 0.8
  assert group_share(disliked, 1) > 0.8

def test_fold_in_uses_the_ratings_not_just_the_movies():

  # The bug this replaces: identical ratings produced identical vectors whatever their value
  rng = np.random.default_rng(1)
  Q, b_i = rng.normal(size=(20, 3)), np.zeros(20)
  _, high = fold_in_user(Q, b_i, 3.5, np.arange(5), np.full(5, 5.0))
  _, low = fold_in_user(Q, b_i, 3.5, np.arange(5), np.full(5, 0.5))
  assert np.dot(high, low) < 0

def test_score_new_user_adds_every_term():
  Q = np.array([[1.0, 0.0], [0.0, 2.0]])
  scores = score_new_user(Q, np.array([0.1, 0.2]), 3.0, 0.5, np.array([1.0, 1.0]))
  assert scores.tolist() == pytest.approx([4.6, 5.7])

def test_new_user_serving_leaves_the_model_unchanged(uv, item, content):
  before = (uv.Q.copy(), item.sim.copy(), content.tfidf_matrix.copy())
  uv.recommend_new_user({1: 5.0, 2: 1.0})
  item.recommend_from_seeds([1, 2])
  content.recommend_from_text('comedy')
  assert (uv.Q == before[0]).all()
  assert (item.sim != before[1]).nnz == 0
  assert (content.tfidf_matrix != before[2]).nnz == 0

def test_serving_copies_work_and_hold_no_ratings(uv, item, content):
  slim_uv, slim_item, slim_content = uv.for_serving(), item.for_serving(), content.for_serving()
  assert slim_uv.P is None and slim_uv.train_df is None
  assert slim_item.r_matrix is None and slim_content.interactions is None
  assert uv.P is not None
  assert len(slim_uv.recommend_new_user({1: 5.0, 3: 4.0}, n=3)) == 3
  assert len(slim_item.recommend_from_seeds([1, 3], n=3)) == 3
  assert len(slim_content.recommend_from_text('comedy', n=3)) == 3

def test_item_neighbours_stay_within_the_taste_group(item):
  row = item.sim[item.movie_idx[2]]
  neighbour_ids = item.movie_ids[row.indices]
  assert len(neighbour_ids) > 0 and (neighbour_ids % 2 == 0).mean() > 0.9

def test_item_seeds_recommend_the_same_group_and_skip_seeds(item):
  seeds = [2, 4, 6]
  result = item.recommend_from_seeds(seeds, n=5)
  assert group_share(result, 0) == 1
  assert not set(seeds) & set(result['movieId'])

def test_item_seeds_unknown_movies_give_nothing(item):
  assert item.recommend_from_seeds([999999]).empty

def test_item_scores_do_not_saturate_into_ties(item, ratings):
  user_id = int(ratings['userId'].iloc[0])
  _, scores = item.score_catalogue(user_id)
  top = np.sort(scores)[-10:]
  assert len(np.unique(top)) == 10

def test_cf_predictions_stay_in_range_and_beat_the_baseline(item, user, ratings, movies):
  bias = BiasBaseline()
  bias.fit(ratings, movies)
  sample = ratings.sample(300, random_state=0)
  users, items, truth = sample['userId'].values, sample['movieId'].values, sample['rating'].values
  bias_error = np.mean((bias.predict(users, items) - truth) ** 2)
  for model in (item, user):
    preds = model.predict(users, items)
    assert preds.min() >= 0.5 and preds.max() <= 5.0
    assert np.mean((preds - truth) ** 2) < bias_error

def test_cf_tracks_how_often_neighbours_are_missing(ratings, movies):
  model = ItemBasedCF(n_neighbors=5, sim_threshold=2.0, block_size=16)
  model.fit(ratings, movies)
  model.reset_fallback_stats()
  model.predict(ratings['userId'].values[:50], ratings['movieId'].values[:50])
  assert model.fallback_rate == 1

def test_unknown_ids_fall_back_without_error(uv, item, user, content):
  for model in (uv, item, user, content):
    assert np.isfinite(model.predict(np.array([999999, 1]), np.array([1, 999999]))).all()

def test_recommend_excludes_rated_movies(uv, ratings):
  user_id = int(ratings['userId'].iloc[0])
  rated = set(ratings.loc[ratings['userId'] == user_id, 'movieId'])
  assert not rated & set(uv.recommend(user_id, n=10)['movieId'])

def test_content_profile_is_centred_on_each_user(content, ratings):

  # Every user's weights sum to zero, whatever scale they rate on
  row_sums = np.asarray(content.interactions.sum(axis=1)).flatten()
  assert np.abs(row_sums).max() < 1e-2

def test_content_genres_are_clean_for_csv_and_list_input(ratings, movies):
  parsed = movies.copy()
  parsed['genres'] = parsed['genres'].str.split('|')
  model = ContentBasedRecommender()
  model.fit(ratings, parsed)
  assert model.genres == ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Thriller']
  assert split_genres('(no genres listed)') == []

def test_content_text_with_no_known_terms_gives_nothing(content):
  assert content.recommend_from_text('zzzzqqq').empty

def test_baselines_rank_as_expected(ratings, movies):
  popular, mean = PopularityBaseline(), ItemMeanBaseline()
  popular.fit(ratings, movies)
  mean.fit(ratings, movies)
  counts = ratings['movieId'].value_counts()
  ids, scores = popular.score_catalogue(1)
  assert ids[np.argmax(scores)] in counts[counts == counts.max()].index
  assert mean.predict(np.array([1]), np.array([999999]))[0] == pytest.approx(ratings['rating'].mean())
