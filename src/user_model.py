# Load Dependencies

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from base_model import BaseRecommender
from baselines import fit_biases

class UserBasedCF(BaseRecommender):

  # Object initialization
  def __init__(self, n_neighbors=100, sim_threshold=0.0, sim_shrinkage=25.0, weight_shrinkage=1.0):
    self.n_neighbors = n_neighbors
    self.sim_threshold = sim_threshold
    self.sim_shrinkage = sim_shrinkage
    self.weight_shrinkage = weight_shrinkage
    self.user_idx = {}
    self.movie_idx = {}
    self.r_matrix = None
    self.rated_matrix = None
    self.r_csc = None
    self.r_norms = None
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

    # Same pattern of 1s, for counting co-rated movies
    self.rated_matrix = self.r_matrix.copy()
    self.rated_matrix.data = np.ones_like(self.rated_matrix.data)

    # Convert to CSC format for efficient column-based movie lookups
    self.r_csc = self.r_matrix.tocsc()

    # Compute user vector magnitudes for cosine similarity
    self.r_norms = np.sqrt(np.asarray(self.r_matrix.multiply(self.r_matrix).sum(axis=1))).flatten()
    self.r_norms[self.r_norms == 0] = 1.0

  def _similarities(self, u_row: int) -> np.ndarray:

    # Cosine between residuals, which compares taste rather than how many movies two users share
    numerator = self.r_matrix.dot(self.r_matrix.getrow(u_row).T).toarray().flatten()
    sims = numerator / (self.r_norms * self.r_norms[u_row])

    # Shrink similarities that rest on few co-rated movies
    co_rated = self.rated_matrix.dot(self.rated_matrix.getrow(u_row).T).toarray().flatten()
    sims *= co_rated / (co_rated + self.sim_shrinkage)

    # A user is not their own neighbour
    sims[u_row] = -1.0
    return sims

  def _adjustment(self, sims: np.ndarray, m_col: int) -> float:

    # Users who rated the movie, read directly from CSC storage
    start, stop = self.r_csc.indptr[m_col], self.r_csc.indptr[m_col + 1]
    peers = self.r_csc.indices[start:stop]
    peer_residuals = self.r_csc.data[start:stop]

    # Keep neighbours above the similarity threshold
    peer_sims = sims[peers]
    valid = peer_sims > self.sim_threshold
    valid_sims = peer_sims[valid]
    self.pairs_predicted += 1

    if len(valid_sims) == 0:
      self.pairs_without_neighbours += 1
      return 0.0

    valid_residuals = peer_residuals[valid]

    # Select nearest neighbours
    if len(valid_sims) > self.n_neighbors:
      top_k = np.argpartition(valid_sims, -self.n_neighbors)[-self.n_neighbors:]
      valid_sims = valid_sims[top_k]
      valid_residuals = valid_residuals[top_k]

    # Shrinkage stops a few neighbours from pushing a prediction to the edge of the scale
    return float(np.dot(valid_sims, valid_residuals) / (valid_sims.sum() + self.weight_shrinkage))

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
      sims = self._similarities(u_row)
      for position in positions.values:
        preds[position] += self._adjustment(sims, m_cols[position])

    return np.clip(preds, 0.5, 5.0)

  def score_catalogue(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
    u_row = self.user_idx[user_id]
    sims = self._similarities(u_row)
    adjustments = np.array([self._adjustment(sims, m_col) for m_col in range(len(self.movie_ids))])
    return self.movie_ids, self.global_mean + self.b_u[u_row] + self.b_i + adjustments
