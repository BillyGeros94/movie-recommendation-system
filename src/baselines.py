# Load Dependencies

from typing import Tuple

import numpy as np
import pandas as pd
from base_model import BaseRecommender

def fit_biases(train_df: pd.DataFrame, damping: float = 10.0) -> Tuple[float, pd.Series, pd.Series]:

  # Damped item then user offsets from the global mean. Damping pulls thinly rated entities towards zero
  global_mean = float(train_df['rating'].mean())
  centred = train_df['rating'].astype('float64') - global_mean

  movie_groups = centred.groupby(train_df['movieId'])
  b_i = movie_groups.sum() / (movie_groups.size() + damping)

  residuals = centred - train_df['movieId'].map(b_i)
  user_groups = residuals.groupby(train_df['userId'])
  b_u = user_groups.sum() / (user_groups.size() + damping)

  return global_mean, b_u, b_i

class _CatalogueBaseline(BaseRecommender):

  # Shared behaviour for baselines whose ranking is the same for every user
  training_only = ('train_df',)

  def _set_item_scores(self, train_df, movies_df, item_scores: pd.Series):
    self._store_catalogue(train_df, movies_df, item_scores.index.values)
    self.item_scores = item_scores.values.astype(np.float64)
    self.item_lookup = item_scores
    self.user_idx = {u: i for i, u in enumerate(train_df['userId'].unique().tolist())}

  def score_catalogue(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
    return self.movie_ids, self.item_scores

class PopularityBaseline(_CatalogueBaseline):

  # Ranks movies by how many ratings they have. It does not estimate ratings
  rating_model = False

  def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame, **kwargs):
    self._set_item_scores(train_df, movies_df, train_df.groupby('movieId').size().astype('float64'))

  def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
    return pd.Series(np.atleast_1d(movie_ids)).map(self.item_lookup).fillna(0.0).values

class ItemMeanBaseline(_CatalogueBaseline):

  def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame, **kwargs):
    self.global_mean = float(train_df['rating'].mean())
    self._set_item_scores(train_df, movies_df, train_df.groupby('movieId')['rating'].mean().astype('float64'))

  def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
    return pd.Series(np.atleast_1d(movie_ids)).map(self.item_lookup).fillna(self.global_mean).values

class BiasBaseline(_CatalogueBaseline):

  # Global mean plus user and item offsets: personalised in scale, not in taste
  def __init__(self, damping: float = 10.0):
    self.damping = damping

  def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame, **kwargs):
    self.global_mean, self.b_u, self.b_i = fit_biases(train_df, self.damping)
    self._set_item_scores(train_df, movies_df, self.global_mean + self.b_i)

  def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
    user_offsets = pd.Series(np.atleast_1d(user_ids)).map(self.b_u).fillna(0.0).values
    movie_offsets = pd.Series(np.atleast_1d(movie_ids)).map(self.b_i).fillna(0.0).values
    return np.clip(self.global_mean + user_offsets + movie_offsets, 0.5, 5.0)

class RandomBaseline(_CatalogueBaseline):

  # The floor every metric should be read against
  rating_model = False

  def __init__(self, seed: int = 42):
    self.rng = np.random.default_rng(seed)

  def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame, **kwargs):
    self._set_item_scores(train_df, movies_df, train_df.groupby('movieId').size().astype('float64'))

  def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
    return self.rng.random(len(np.atleast_1d(user_ids)))

  def score_catalogue(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
    return self.movie_ids, self.rng.random(len(self.movie_ids))
