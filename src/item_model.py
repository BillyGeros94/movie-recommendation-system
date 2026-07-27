import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from base_model import BaseRecommender

class ItemBasedCF(BaseRecommender):

  # Object initialization
  def __init__(self, n_neighbors=100, sim_threshold=0.3):
    self.n_neighbors = n_neighbors
    self.sim_threshold = sim_threshold
    self.user_idx = {}
    self.user_inv_idx = {}
    self.movie_idx = {}
    self.movie_inv_idx = {}
    self.r_matrix = None
    self.b_matrix = None
    self.movie_norms = None
    self.r_csc = None
    self.user_means = None
    self.movie_means = None
    self.global_mean = 3.5
    self.train_df = None
    self.movies_df = None

  def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame):

    # Store training and movie data
    self.train_df = train_df
    self.movies_df = movies_df

    # Compute global, user and movie mean ratings
    self.global_mean = float(train_df['rating'].mean())
    self.user_means = train_df.groupby('userId')['rating'].mean().to_dict()
    self.movie_means = train_df.groupby('movieId')['rating'].mean().to_dict()

    # Extract unique users and movies
    users = train_df['userId'].unique()
    movies = train_df['movieId'].unique()

    # Create mappings between IDs and matrix indices
    self.user_idx = {u: i for i, u in enumerate(users)}
    self.user_inv_idx = {i: u for u, i in self.user_idx.items()}
    self.movie_idx = {m: i for i, m in enumerate(movies)}
    self.movie_inv_idx = {i: m for m, i in self.movie_idx.items()}

    # Extract userId, movieId and rating values
    rows = train_df['userId'].map(self.user_idx).values
    cols = train_df['movieId'].map(self.movie_idx).values
    vals = train_df['rating'].values

    n_users, n_movies = len(users), len(movies)

    # Build user-item rating matrix
    self.r_matrix = csr_matrix((vals, (rows, cols)), shape=(n_users, n_movies))

    # Convert to CSC format for efficient column-based movie lookups
    self.r_csc = self.r_matrix.tocsc()

    # Mean-centre ratings using movie mean ratings
    m_means_mapped = train_df['movieId'].map(self.movie_means).values
    centered_vals = vals - m_means_mapped
    self.b_matrix = csr_matrix((centered_vals, (rows, cols)), shape=(n_users, n_movies))

    # Compute movie vector magnitudes for cosine similarity
    self.movie_norms = np.sqrt(np.asarray(self.r_matrix.power(2).sum(axis=0))).flatten()
    self.movie_norms[self.movie_norms == 0] = 1.0

  def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:

    # Convert userId and movieId to 1D arrays
    user_ids = np.atleast_1d(user_ids)
    movie_ids = np.atleast_1d(movie_ids)

    # Fill preds with the global mean
    preds = np.full(len(user_ids), self.global_mean)

    # Map userId and movieId to internal matrix indices
    u_indices = pd.Series(user_ids).map(self.user_idx).fillna(-1).astype(int).values
    m_indices = pd.Series(movie_ids).map(self.movie_idx).fillna(-1).astype(int).values

    # Store mapped indices while preserving original userId, movieId and prediction order
    eval_df = pd.DataFrame({
      'u_row': u_indices, 
      'm_col': m_indices, 
      'orig_u': user_ids,
      'orig_m': movie_ids, 
      'idx': np.arange(len(user_ids))
    })
        
    for u_row, group in eval_df.groupby('u_row'):

      # Check if user found
      if u_row == -1:
        preds[group['idx'].values] = [self.movie_means.get(m, self.global_mean) for m in group['orig_m'].values]
        continue

      # Retrieve movies rated by the user from the CSR rating matrix        
      idx_start = self.r_matrix.indptr[u_row]
      idx_end = self.r_matrix.indptr[u_row+1]
      rated_movie_cols = self.r_matrix.indices[idx_start:idx_end]

      # Require enough rated movies
      if len(rated_movie_cols) == 0:
        orig_u = self.user_inv_idx[u_row]
        preds[group['idx'].values] = [self.movie_means.get(m, self.user_means.get(orig_u, self.global_mean)) for m in group['orig_m'].values]
        continue

      # Current user's mean-centred ratings
      u_centered = np.asarray(self.b_matrix.data[idx_start:idx_end])

      # Extract movie indices, prediction positions and original movieId
      g_m_cols = group['m_col'].values
      g_idxs = group['idx'].values
      g_orig_m = group['orig_m'].values

      # Keep only movies that exist in the training data
      valid_m_mask = g_m_cols != -1
      unique_requested_cols = np.unique(g_m_cols[valid_m_mask])

      # Fall back if none of the requested movies exist in the model
      if len(unique_requested_cols) == 0:
        preds[g_idxs] = [self.movie_means.get(m, self.global_mean) for m in g_orig_m]
        continue

      # Compute cosine similarity between candidate movies and the user's rated movies 
      r_watched = self.r_csc[:, rated_movie_cols]
      numerators = self.r_csc[:, unique_requested_cols].T.dot(r_watched).toarray()
      denominators = self.movie_norms[unique_requested_cols, np.newaxis] * self.movie_norms[rated_movie_cols]

      # Clip cosine similarity scores to the range [0.0, 1.0]
      all_sims = np.clip(numerators / denominators, 0.0, 1.0)
      all_sims[np.isnan(all_sims) | (all_sims <= self.sim_threshold)] = 0.0

      # Select nearest neighbors
      if all_sims.shape[1] > self.n_neighbors:
        k_bounds = np.partition(all_sims, -self.n_neighbors, axis=1)[:, -self.n_neighbors][:, np.newaxis]
        all_sims[all_sims < k_bounds] = 0.0

      # Compute similarity-weighted neighbour contribution and normalization factor
      num_sum = all_sims.dot(u_centered)
      den_sum = np.sum(all_sims, axis=1)
      sim_map = {col: i for i, col in enumerate(unique_requested_cols)}

      # Predict rating using weighted sum of neighbours' mean-centred ratings
      for m_col, idx, orig_m in zip(g_m_cols, g_idxs, g_orig_m):
        if m_col == -1:
          preds[idx] = self.movie_means.get(orig_m, self.global_mean)
        else:
          row_idx = sim_map[m_col]
          m_mean = self.movie_means[orig_m]
          if den_sum[row_idx] > 0:
            preds[idx] = np.clip(m_mean + (num_sum[row_idx] / den_sum[row_idx]), 0.5, 5.0)
          else:
            preds[idx] = m_mean
                    
    return preds

  def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:

    # Return empty if user not found
    if user_id not in self.user_idx:
      return pd.DataFrame(columns=['title', 'predicted_rating'])

    # Filter for unrated movies
    u_row = self.user_idx[user_id]
    idx_start = self.r_matrix.indptr[u_row]
    idx_end = self.r_matrix.indptr[u_row+1]
    rated_movie_cols = self.r_matrix.indices[idx_start:idx_end]
    rated_movie_ids = {self.movie_inv_idx[i] for i in rated_movie_cols}
    unrated_movie_ids = np.array([m for m in self.movie_idx if m not in rated_movie_ids])

    # Predict ratings
    target_user_ids = np.full(len(unrated_movie_ids), user_id)
    preds = self.predict(target_user_ids, unrated_movie_ids)

    # Construct recommendations df and filter top n recommendations
    recommendations = pd.DataFrame({'movieId': unrated_movie_ids, 'predicted_rating': preds})
    top_df = recommendations.nlargest(n, 'predicted_rating')

    # Add movie title
    result = top_df.merge(self.movies_df[['movieId', 'title']], on='movieId')[['title', 'predicted_rating']]
    result['predicted_rating'] = result['predicted_rating'].round(3)

    return result.reset_index(drop=True)