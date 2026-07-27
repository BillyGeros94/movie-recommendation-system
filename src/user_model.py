# Load Dependencies

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from base_model import BaseRecommender

class UserBasedCF(BaseRecommender):

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
    self.r_norms = None
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
    self.user_means = train_df.groupby('userId')['rating'].mean()
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

    # Mean-centre ratings by user means
    u_means_mapped = train_df['userId'].map(self.user_means).values
    centered_vals = vals - u_means_mapped
    self.b_matrix = csr_matrix((centered_vals, (rows, cols)), shape=(n_users, n_movies))

    # Compute user vector magnitudes for cosine similarity
    self.r_norms = np.sqrt(np.asarray(self.r_matrix.power(2).sum(axis=1))).flatten()
    self.r_norms[self.r_norms == 0] = 1.0

  def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:

    # Convert userId and movieId to 1D arrays
    user_ids = np.atleast_1d(user_ids)
    movie_ids = np.atleast_1d(movie_ids)

    # Fill preds with the global mean
    preds = np.full(len(user_ids), self.global_mean)

    # Map userId and movieId to internal matrix indices
    u_indices = pd.Series(user_ids).map(self.user_idx).fillna(-1).astype(int).values
    m_indices = pd.Series(movie_ids).map(self.movie_idx).fillna(-1).astype(int).values

    # Store mapped indices while preserving original movieId and prediction order
    eval_df = pd.DataFrame({
      'u_row': u_indices, 
      'm_col': m_indices, 
      'orig_m': movie_ids, 
      'idx': np.arange(len(user_ids))
    })

    for u_row, group in eval_df.groupby('u_row'):

      # Check if user found
      if u_row == -1:
        preds[group['idx'].values] = [self.movie_means.get(m, self.global_mean) for m in group['orig_m'].values]
        continue

      # Retrieve original userId and average rating
      orig_u = self.user_inv_idx[u_row]
      u_mean = self.user_means[orig_u]

      # Get user vector
      r_u = self.r_matrix.getrow(u_row)

      # Compute cosine similarity between target user and all other users 
      numerator = self.r_matrix.dot(r_u.T).toarray().flatten()
      denominator = self.r_norms * self.r_norms[u_row]

      # Clip cosine similarity scores to the range [0.0, 1.0]
      all_sims = np.clip(numerator / denominator, 0.0, 1.0)
      all_sims[u_row] = -1.0

      # Extract movie indices, prediction positions and original movieId
      g_m_cols = group['m_col'].values
      g_idxs = group['idx'].values
      g_orig_m = group['orig_m'].values

      
      for m_col, idx, orig_m in zip(g_m_cols, g_idxs, g_orig_m):

        # Check if movie found
        if m_col == -1:
          preds[idx] = u_mean
          continue

        # Retrieve users who rated the movie using CSC column indexing
        idx_start = self.r_csc.indptr[m_col]
        idx_end = self.r_csc.indptr[m_col+1]
        peers_who_rated = self.r_csc.indices[idx_start:idx_end]

        # Require enough users who rated the movie   
        if len(peers_who_rated) < 3:
          preds[idx] = self.movie_means.get(orig_m, u_mean)
          continue

        # Similarity threshold
        peer_sims = all_sims[peers_who_rated]
        valid_mask = (peer_sims > self.sim_threshold)

        # Require enough similar neighbours
        if valid_mask.sum() < 3:
          preds[idx] = self.movie_means.get(orig_m, u_mean)
          continue

        # Keep neighbours above similarity threshold
        valid_sims = peer_sims[valid_mask]
        valid_peer_rows = peers_who_rated[valid_mask]

        # Select nearest neighbors  
        if len(valid_sims) > self.n_neighbors:
          top_k = np.argpartition(valid_sims, -self.n_neighbors)[-self.n_neighbors:]
          valid_sims = valid_sims[top_k]
          valid_peer_rows = valid_peer_rows[top_k]

        # Obtain neighbours' mean-centred ratings
        centered_ratings = np.asarray(self.b_matrix[valid_peer_rows, m_col].toarray()).flatten()

        # Compute similarity-weighted neighbour contribution and normalization factor
        num_sum = np.dot(valid_sims, centered_ratings)
        den_sum = np.sum(np.abs(valid_sims))

        # Predict rating using weighted sum of neighbours' mean-centred ratings 
        if den_sum == 0:
          preds[idx] = self.movie_means.get(orig_m, u_mean)
        else:
          preds[idx] = np.clip(u_mean + (num_sum / den_sum), 0.5, 5.0)
                    
    return preds

  def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:

    # Return empty if user not found
    if user_id not in self.user_idx:
      return pd.DataFrame(columns=['title', 'predicted_rating'])

    # Filter for unrated movies
    rated_movies = set(self.train_df[self.train_df['userId'] == user_id]['movieId'])
    unrated_movie_ids = np.array([m for m in self.movie_idx if m not in rated_movies])

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