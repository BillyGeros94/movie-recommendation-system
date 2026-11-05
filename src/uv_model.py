# Load Dependencies

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from base_model import BaseRecommender
from fold_in import fold_in_user, score_new_user

class UVDecomposition(BaseRecommender):

  training_only = ('train_df', 'P', 'b_u', 'user_idx')

  # Object initialization
  def __init__(self, n_factors=50, std=0.05, seed=42, fold_in_reg=2.0):
    self.n_factors = n_factors
    self.std = std
    self.seed = seed
    self.fold_in_reg = fold_in_reg
    self.user_idx = {}
    self.movie_idx = {}
    self.P = None
    self.Q = None
    self.train_df = None
    self.b_u = None
    self.b_i = None
    self.global_mean = 3.5
    self.history = []
    self.best_epoch = None

  def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame, n_epochs=50, lr=0.005, reg=0.02,
          batch_size=20000, val_df: Optional[pd.DataFrame] = None, patience=3):

    # Compute global mean rating
    self.global_mean = float(train_df['rating'].mean())

    # Extract unique users and movies
    users = train_df['userId'].unique()
    movies = train_df['movieId'].unique()
    self._store_catalogue(train_df, movies_df, movies)

    # Create mappings between IDs and matrix indices
    self.user_idx = {u: i for i, u in enumerate(users.tolist())}
    self.movie_idx = {m: i for i, m in enumerate(movies.tolist())}

    rng = np.random.default_rng(self.seed)

    # Zero-mean factors: the biases carry the rating level, so P.Q should start near zero
    self.Q = rng.normal(0.0, self.std, size=(len(movies), self.n_factors)).astype(np.float32)
    self.P = rng.normal(0.0, self.std, size=(len(users), self.n_factors)).astype(np.float32)

    # Initialise biases
    self.b_u = np.zeros(len(users), dtype=np.float32)
    self.b_i = np.zeros(len(movies), dtype=np.float32)

    # Extract user, movieId and rating values
    user_indices = train_df['userId'].map(self.user_idx).values.astype(np.int32)
    movie_indices = train_df['movieId'].map(self.movie_idx).values.astype(np.int32)
    ratings = train_df['rating'].values.astype(np.float32)
    num_observations = len(ratings)

    self.history = []
    self.best_epoch = None
    best_rmse, best_state = np.inf, None

    # Training loop
    for epoch in range(1, n_epochs + 1):

      # Randomly shuffle observations each epoch
      shuffled_idx = rng.permutation(num_observations).astype(np.int32)
      epoch_squared_errors = 0.0

      for start_idx in range(0, num_observations, batch_size):

        # Extract batch data
        batch_idx = shuffled_idx[start_idx:start_idx + batch_size]
        u_batch = user_indices[batch_idx]
        m_batch = movie_indices[batch_idx]

        P_batch = self.P[u_batch]
        Q_batch = self.Q[m_batch]

        # Compute prediction errors
        batch_errors = (ratings[batch_idx] - np.float32(self.global_mean) - self.b_u[u_batch] - self.b_i[m_batch]
                        - np.einsum('ij,ij->i', P_batch, Q_batch))
        epoch_squared_errors += float(np.dot(batch_errors, batch_errors))

        # Gradients for both factor matrices, clipped because one batch can repeat a popular movie many times
        p_grads = np.clip(batch_errors[:, np.newaxis] * Q_batch - reg * P_batch, -5.0, 5.0)
        q_grads = np.clip(batch_errors[:, np.newaxis] * P_batch - reg * Q_batch, -5.0, 5.0)

        # Update latent vectors
        np.add.at(self.P, u_batch, lr * p_grads)
        np.add.at(self.Q, m_batch, lr * q_grads)

        # Bias updates
        np.add.at(self.b_u, u_batch, lr * (batch_errors - reg * self.b_u[u_batch]))
        np.add.at(self.b_i, m_batch, lr * (batch_errors - reg * self.b_i[m_batch]))

      record = {'epoch': epoch, 'train_rmse': round(float(np.sqrt(epoch_squared_errors / num_observations)), 4)}

      # Early stopping watches held-out ratings, so it stops at overfitting rather than at a fixed training error
      if val_df is not None:
        record['val_rmse'] = round(self._rmse(val_df), 4)
        if record['val_rmse'] < best_rmse:
          best_rmse = record['val_rmse']
          self.best_epoch = epoch
          best_state = (self.P.copy(), self.Q.copy(), self.b_u.copy(), self.b_i.copy())

      self.history.append(record)
      print(record, flush=True)

      if val_df is not None and epoch - self.best_epoch >= patience:
        break

    # Keep the parameters from the best validation epoch
    if best_state is not None:
      self.P, self.Q, self.b_u, self.b_i = best_state

  def _rmse(self, ratings_df: pd.DataFrame) -> float:
    errors = ratings_df['rating'].values - self.predict(ratings_df['userId'].values, ratings_df['movieId'].values)
    return float(np.sqrt(np.mean(errors ** 2)))

  def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:

    # Convert userId and movieId to 1D arrays
    user_ids = np.atleast_1d(user_ids)
    movie_ids = np.atleast_1d(movie_ids)

    # Fill preds with the global mean
    preds = np.full(len(user_ids), self.global_mean)

    # Map userId and movieId to internal matrix indices
    u_idx = pd.Series(user_ids).map(self.user_idx).fillna(-1).astype(int).values
    m_idx = pd.Series(movie_ids).map(self.movie_idx).fillna(-1).astype(int).values

    # Keep only users and movies seen during training
    valid = (u_idx >= 0) & (m_idx >= 0)

    # Predict ratings using latent vectors
    preds[valid] = np.clip(self.global_mean + self.b_u[u_idx[valid]] + self.b_i[m_idx[valid]]
      + np.einsum('ij,ij->i', self.Q[m_idx[valid]], self.P[u_idx[valid]]), 0.5, 5.0)

    return preds

  def score_catalogue(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
    user_row = self.user_idx[user_id]
    return self.movie_ids, self.global_mean + self.b_u[user_row] + self.b_i + self.Q @ self.P[user_row]

  def recommend_new_user(self, ratings: Dict[int, float], n: int = 10) -> pd.DataFrame:

    # Serve a user the model never saw, without changing model state
    known = {m: r for m, r in ratings.items() if m in self.movie_idx}
    if not known:
      return pd.DataFrame(columns=['movieId', 'title', 'predicted_rating'])

    item_indices = np.array([self.movie_idx[m] for m in known])
    b_u, p_u = fold_in_user(self.Q, self.b_i, self.global_mean, item_indices,
                            np.array(list(known.values())), reg=self.fold_in_reg)
    scores = score_new_user(self.Q, self.b_i, self.global_mean, b_u, p_u)

    return self._top_n(self.movie_ids, scores, set(ratings), n)
