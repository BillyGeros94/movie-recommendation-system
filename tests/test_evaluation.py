import numpy as np
import pandas as pd
import pytest

from baselines import PopularityBaseline, RandomBaseline
from evaluation import RecommenderEvaluator

class Oracle:
  rating_model = True

  def __init__(self, test):
    self.lookup = {(u, m): r for u, m, r in zip(test['userId'], test['movieId'], test['rating'])}

  def predict(self, user_ids, movie_ids):
    return np.array([self.lookup[(u, m)] for u, m in zip(user_ids.tolist(), movie_ids.tolist())])

class FixedRanking:

  # Scores movies by a fixed order, the same for every user
  def __init__(self, movie_ids):
    self.movie_ids = np.array(movie_ids)

  def score_catalogue(self, user_id):
    return self.movie_ids, -np.arange(len(self.movie_ids), dtype=float)

@pytest.fixture
def test_df():
  return pd.DataFrame({
    'userId': [1] * 4 + [2] * 4,
    'movieId': [1, 2, 3, 4] * 2,
    'rating': [5.0, 4.0, 2.0, 1.0, 1.0, 2.0, 4.0, 5.0]
  })

def test_perfect_predictions_score_perfectly(test_df):
  results = RecommenderEvaluator(test_df).full_evaluation(Oracle(test_df), n=2)
  assert results['RMSE'] == 0
  assert results['Precision@2'] == 1
  assert results['Recall@2'] == 1
  assert results['nDCG@2'] == 1

def test_ranking_models_report_no_rating_error(test_df, ratings, movies):
  model = PopularityBaseline()
  model.fit(ratings, movies)
  results = RecommenderEvaluator(test_df).full_evaluation(model, n=2)
  assert results['RMSE'] is None and results['MAE'] is None

def test_catalogue_metrics_count_hits_and_skip_seen_movies():
  test = pd.DataFrame({'userId': [1, 1], 'movieId': [3, 4], 'rating': [5.0, 2.0]})
  train = pd.DataFrame({'userId': [1, 2, 2], 'movieId': [1, 1, 2], 'rating': [4.0, 4.0, 4.0]})

  # Movie 1 ranks first but is already seen, so the top 2 are movies 2 and 3. Only 3 is liked
  results = RecommenderEvaluator(test).catalogue_evaluation(FixedRanking([1, 2, 3, 4]), train, n=2)
  assert results['Catalogue P@2'] == 0.5
  assert results['Catalogue R@2'] == 1.0
  assert results['Catalogue nDCG@2'] == pytest.approx(1 / np.log2(3), abs=1e-4)

def test_random_order_is_far_from_perfect_on_the_catalogue(ratings, movies):
  liked = ratings[ratings['rating'] >= 3.5].groupby('userId').head(2)
  train = ratings.drop(liked.index)
  model = RandomBaseline()
  model.fit(train, movies)
  results = RecommenderEvaluator(liked).catalogue_evaluation(model, train, n=5)
  assert results['Catalogue R@5'] < 1
  assert results['Coverage'] > 0.5
