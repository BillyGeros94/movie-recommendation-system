# Load Dependencies

import copy
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd

class BaseRecommender(ABC):

  # False for models that rank but do not estimate a rating (e.g. popularity)
  rating_model = True

  # Attributes needed only for training or known-user lookups, dropped by for_serving()
  training_only: Tuple[str, ...] = ()

  @abstractmethod
  def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame, **kwargs):
    pass

  @abstractmethod
  def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
    pass

  @abstractmethod
  def score_catalogue(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (movie_ids, scores) for every movie in the model. Scores are not clipped, so they keep their order."""

  def _store_catalogue(self, train_df: pd.DataFrame, movies_df: pd.DataFrame, movie_ids: np.ndarray):

    # Keep the data needed to turn scores into titled recommendations
    self.train_df = train_df
    self.movie_ids = np.asarray(movie_ids)
    known = movies_df[movies_df['movieId'].isin(self.movie_ids)]
    self.titles = dict(zip(known['movieId'].tolist(), known['title'].tolist()))

  def _top_n(self, movie_ids: np.ndarray, scores: np.ndarray, exclude, n: int) -> pd.DataFrame:

    # Rank unseen movies on unclipped scores, so saturated predictions do not tie
    keep = ~np.isin(movie_ids, list(exclude))
    candidates, candidate_scores = movie_ids[keep], scores[keep]
    top = np.argsort(-candidate_scores, kind='stable')[:n]
    top = top[np.isfinite(candidate_scores[top])]

    return pd.DataFrame({
      'movieId': candidates[top],
      'title': [self.titles.get(int(m), str(m)) for m in candidates[top]],
      'predicted_rating': np.clip(candidate_scores[top], 0.5, 5.0).round(3)
    })

  def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:

    # Return empty if user not found
    if user_id not in self.user_idx:
      return pd.DataFrame(columns=['movieId', 'title', 'predicted_rating'])

    movie_ids, scores = self.score_catalogue(user_id)
    rated_movies = set(self.train_df.loc[self.train_df['userId'] == user_id, 'movieId'].tolist())

    return self._top_n(movie_ids, scores, rated_movies, n)

  def for_serving(self):

    # Copy without training-only state, for small artifacts that hold no ratings
    slim = copy.copy(self)
    for attribute in self.training_only:
      setattr(slim, attribute, None)
    return slim
