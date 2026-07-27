import numpy as np
import pandas as pd
from base_model import BaseRecommender

class UVDecomposition(BaseRecommender):

  # Object initialization
  def __init__(self, n_factors=15, std=0.05, seed=42):
    self.n_factors = n_factors
    self.std = std
    self.seed = seed
    self.user_idx = {}
    self.movie_idx = {}
    self.P = None
    self.Q = None
    self.train_df = None
    self.movies_df = None
    self.global_mean = 3.5

  def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame, n_epochs=50, lr=0.001, reg=0.02, target_mse=0.55, batch_size=100000):

    # Store training and movie data
    self.train_df = train_df
    self.movies_df = movies_df

    # Compute global mean rating 
    self.global_mean = float(train_df['rating'].mean())

    # Extract unique users and movies
    users = train_df['userId'].unique()
    movies = train_df['movieId'].unique()

    # Create mappings between IDs and matrix indices
    self.user_idx = {u: i for i, u in enumerate(users)}
    self.movie_idx = {m: i for i, m in enumerate(movies)}

     # Mean used for latent vector initialisation
    mean_init = np.sqrt(1.0 / self.n_factors)
    np.random.seed(self.seed)

    # Initialise movie latent factor matrix (Q)
    self.Q = np.random.normal(loc=mean_init, scale=self.std, size=(len(movies), self.n_factors))

    # Initialise user latent factor matrix (P)
    self.P = np.random.normal(loc=mean_init, scale=self.std, size=(len(users), self.n_factors))

    # Extract user, movieId and rating values
    user_indices = train_df['userId'].map(self.user_idx).values
    movie_indices = train_df['movieId'].map(self.movie_idx).values
    ratings = train_df['rating'].values
    num_observations = len(ratings)

    # Training loop
    for epoch in range(n_epochs):

      # Randomly shuffle observations each epoch
      shuffled_idx = np.arange(num_observations)
      np.random.shuffle(shuffled_idx)

      epoch_squared_errors = 0

      for start_idx in range(0, num_observations, batch_size):

        # Set batch boundaries
        end_idx = min(start_idx + batch_size, num_observations)
        batch_idx = shuffled_idx[start_idx:end_idx]

        # Extract batch data
        u_batch = user_indices[batch_idx]
        m_batch = movie_indices[batch_idx]
        r_batch = ratings[batch_idx]

        # Predict ratings using dot product of latent vectors
        batch_preds = np.sum(self.Q[m_batch] * self.P[u_batch], axis=1)

        # Compute prediction errors
        batch_errors = r_batch - batch_preds
        epoch_squared_errors += np.sum(batch_errors ** 2)

        # Compute gradients for users
        p_grads = batch_errors[:, np.newaxis] * self.Q[m_batch] - reg * self.P[u_batch]

        # Prevent exploding gradients
        p_grads = np.clip(p_grads, -5.0, 5.0)

        # Update user latent vectors
        for i in range(self.n_factors):
          np.add.at(self.P[:, i], u_batch, lr * p_grads[:, i])

        # Compute gradients for movies
        q_grads = batch_errors[:, np.newaxis] * self.P[u_batch] - reg * self.Q[m_batch]

        # Prevent exploding gradients
        q_grads = np.clip(q_grads, -5.0, 5.0)

        # Update movie latent vectors
        for i in range(self.n_factors):
            np.add.at(self.Q[:, i], m_batch, lr * q_grads[:, i])

      # Compute Mean Squared Error for the epoch
      mse = epoch_squared_errors / num_observations
      print(f"Epoch {epoch+1:02d}/{n_epochs}, observed MSE: {mse:.4f}")

      # Stop early if desired accuracy is reached
      if mse <= target_mse:
        break

    print(f"Training complete. Final MSE: {mse:.4f}")

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
    preds[valid] = np.clip(np.sum(self.Q[m_idx[valid]] * self.P[u_idx[valid]], axis=1), 0.5, 5.0)
    return preds

  def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:

    # Return empty if user not found
    if user_id not in self.user_idx:
      return pd.DataFrame(columns=['title', 'predicted_rating'])

    # Get user's latent vector index
    user_row = self.user_idx[user_id]

    # Movies already rated by the user
    rated_movies = self.train_df[self.train_df['userId'] == user_id]['movieId']

    # Predict ratings for every movie
    all_preds = np.clip(np.dot(self.Q, self.P[user_row]), 0.5, 5.0)

    # Recover original movieId
    movie_ids = list(self.movie_idx.keys())
    movie_cols = list(self.movie_idx.values())

    # Construct recommendations df
    recommendations = pd.DataFrame({
      'movieId': movie_ids,
      'predicted_rating': all_preds[movie_cols]
    })

    # Filter for unrated movies
    unrated = recommendations[~recommendations['movieId'].isin(rated_movies)]

    # Get top n recommendations
    top_df = unrated.nlargest(n, 'predicted_rating')
    result = top_df.merge(self.movies_df[['movieId', 'title']], on='movieId')[['title', 'predicted_rating']]
    result['predicted_rating'] = result['predicted_rating'].round(3)

    return result.reset_index(drop=True)