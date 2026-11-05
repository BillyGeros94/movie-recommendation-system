# Load Dependencies

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from base_model import BaseRecommender
from baselines import fit_biases

class ItemBasedCF(BaseRecommender):

  training_only = ('train_df', 'r_matrix', 'b_u', 'user_idx')

  # Object initialization
  def __init__(self, n_neighbors=100, sim_threshold=0.0, sim_shrinkage=100.0, weight_shrinkage=1.0, block_size=512):
    self.n_neighbors = n_neighbors
    self.sim_threshold = sim_threshold
    self.sim_shrinkage = sim_shrinkage
    self.weight_shrinkage = weight_shrinkage
    self.block_size = block_size
    self.user_idx = {}
    self.movie_idx = {}
    self.r_matrix = None
    self.sim = None
    self.b_u = None
    self.b_i = None
    self.global_mean = 3.5
    self.train_df = None
    self.reset_fallback_stats()

  def reset_fallback_stats(self):
    self.pairs_predicted = 0
    self.pairs_without_neighbours = 0

  @property
  def fallback_rate(self) -> float:
    return self.pairs_without_neighbours / max(self.pairs_predicted, 1)

  def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame):

    # Extract unique users and movies
    users = train_df['userId'].unique()
    movies = train_df['movieId'].unique()
    self._store_catalogue(train_df, movies_df, movies)

    # Create mappings between IDs and matrix indices
    self.user_idx = {u: i for i, u in enumerate(users.tolist())}
    self.movie_idx = {m: i for i, m in enumerate(movies.tolist())}

    # Baseline per user and movie: similarities and predictions work on what it leaves unexplained
    self.global_mean, user_bias, movie_bias = fit_biases(train_df, damping=10.0)
    self.b_u = user_bias.reindex(users).values.astype(np.float32)
    self.b_i = movie_bias.reindex(movies).values.astype(np.float32)

    # Extract user and movie indices
    rows = train_df['userId'].map(self.user_idx).values.astype(np.int32)
    cols = train_df['movieId'].map(self.movie_idx).values.astype(np.int32)

    # Build the sparse matrix of baseline residuals
    residuals = (train_df['rating'].values - self.global_mean - self.b_u[rows] - self.b_i[cols]).astype(np.float32)
    self.r_matrix = csr_matrix((residuals, (rows, cols)), shape=(len(users), len(movies)))

    self.sim = self._item_neighbours()

  def _item_neighbours(self) -> csr_matrix:

    # Item-major copies make each block of movies a cheap slice
    items = self.r_matrix.T.tocsr()
    rated_items = items.copy()
    rated_items.data = np.ones_like(rated_items.data)
    rated_users = self.r_matrix.copy()
    rated_users.data = np.ones_like(rated_users.data)

    norms = np.sqrt(np.asarray(items.multiply(items).sum(axis=1))).flatten()
    norms[norms == 0] = 1.0

    n_movies = len(self.movie_ids)
    blocks = []

    # Similarity is computed in blocks, so the full item-item matrix never exists in memory
    for start in range(0, n_movies, self.block_size):
      stop = min(start + self.block_size, n_movies)

      # Adjusted cosine between residuals, shrunk when few users rated both movies
      block = (items[start:stop] @ self.r_matrix).toarray() / (norms[start:stop, np.newaxis] * norms)
      co_rated = (rated_items[start:stop] @ rated_users).toarray()
      block *= co_rated / (co_rated + self.sim_shrinkage)

      # A movie is not its own neighbour, and only positive similarity is used
      block[np.arange(stop - start), np.arange(start, stop)] = 0.0
      block[block <= self.sim_threshold] = 0.0

      # Keep each movie's strongest neighbours
      if self.n_neighbors < n_movies:
        cutoff = np.partition(block, -self.n_neighbors, axis=1)[:, -self.n_neighbors][:, np.newaxis]
        block[block < cutoff] = 0.0

      blocks.append(csr_matrix(block.astype(np.float32)))
      print(f'Item similarities: {stop}/{n_movies}', flush=True)

    return vstack(blocks).tocsr()

  def _adjustment(self, candidate_cols: Optional[np.ndarray], rated_cols: np.ndarray, residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    # Similarity-weighted residual of the rated movies that neighbour each candidate
    residual_vector = np.zeros(len(self.movie_ids), dtype=np.float32)
    residual_vector[rated_cols] = residuals
    rated_vector = np.zeros(len(self.movie_ids), dtype=np.float32)
    rated_vector[rated_cols] = 1.0

    neighbours = self.sim if candidate_cols is None else self.sim[candidate_cols]
    numerators = neighbours @ residual_vector
    weights = neighbours @ rated_vector

    # Shrinkage stops one strong neighbour from pushing a prediction to the edge of the scale
    return numerators / (weights + self.weight_shrinkage), weights

  def _user_ratings(self, u_row: int) -> Tuple[np.ndarray, np.ndarray]:
    start, stop = self.r_matrix.indptr[u_row], self.r_matrix.indptr[u_row + 1]
    return self.r_matrix.indices[start:stop], self.r_matrix.data[start:stop]

  def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:

    # Convert userId and movieId to 1D arrays
    user_ids = np.atleast_1d(user_ids)
    movie_ids = np.atleast_1d(movie_ids)

    # Map userId and movieId to internal matrix indices
    u_rows = pd.Series(user_ids).map(self.user_idx).fillna(-1).astype(int).values
    m_cols = pd.Series(movie_ids).map(self.movie_idx).fillna(-1).astype(int).values

    # Start from the baseline, using whichever offsets are known
    preds = np.full(len(user_ids), self.global_mean)
    preds[u_rows >= 0] += self.b_u[u_rows[u_rows >= 0]]
    preds[m_cols >= 0] += self.b_i[m_cols[m_cols >= 0]]

    # Add the neighbourhood adjustment, one user at a time
    known = np.flatnonzero((u_rows >= 0) & (m_cols >= 0))
    for u_row, positions in pd.Series(known).groupby(u_rows[known]):
      positions = positions.values
      rated_cols, residuals = self._user_ratings(u_row)
      adjustment, weights = self._adjustment(m_cols[positions], rated_cols, residuals)
      preds[positions] += adjustment
      self.pairs_predicted += len(positions)
      self.pairs_without_neighbours += int((weights == 0).sum())

    return np.clip(preds, 0.5, 5.0)

  def score_catalogue(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
    u_row = self.user_idx[user_id]
    rated_cols, residuals = self._user_ratings(u_row)
    adjustment, _ = self._adjustment(None, rated_cols, residuals)
    return self.movie_ids, self.global_mean + self.b_u[u_row] + self.b_i + adjustment

  def recommend_from_seeds(self, seed_movie_ids: Iterable[int], n: int = 10) -> pd.DataFrame:

    # Serve "more like these" for a new user, treating each seed as a 5-star rating, without changing model state
    seeds = [m for m in seed_movie_ids if m in self.movie_idx]
    if not seeds:
      return pd.DataFrame(columns=['movieId', 'title', 'predicted_rating'])

    seed_cols = np.array([self.movie_idx[m] for m in seeds])
    residuals = 5.0 - self.global_mean - self.b_i[seed_cols]
    adjustment, weights = self._adjustment(None, seed_cols, residuals)
    scores = self.global_mean + self.b_i + adjustment

    # Only movies related to a seed qualify, otherwise this just returns the highest rated movies
    scores = np.where(weights > 0, scores, -np.inf)
    return self._top_n(self.movie_ids, scores, set(seeds), n)
