# Load Dependencies

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from base_model import BaseRecommender

class ContentBasedRecommender(BaseRecommender):

  # Object initialization
  def __init__(self):
    self.vectorizer = TfidfVectorizer(max_df=0.8, stop_words='english')
    self.user_profiles_matrix = None
    self.user_profiles_norms = None
    self.user_idx_map = {}
    self.tfidf_matrix = None
    self.tfidf_norms = None
    self.movie_idx = {}
    self.movies_df = None
    self.train_df = None
    self.global_mean = 3.5

  def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame, tags_df: pd.DataFrame = None, scores_df: pd.DataFrame = None):

    # Store training and movie data
    self.train_df = train_df
    self.movies_df = movies_df.copy()

    # Compute global mean rating 
    self.global_mean = float(train_df['rating'].mean())

    # Prepare genre metadata 
    self.movies_df['genres_str'] = self.movies_df['genres'].fillna('').astype('str').apply(
      lambda x: ' '.join(x) if isinstance(x, list) else str(x).replace('|', ' ')
    )

    if tags_df is not None and scores_df is not None:
      # Merge tag metadata with relevance scores and keep strong tags
      merged_tags = pd.merge(scores_df, tags_df, on='tagId')
      strong_tags = merged_tags[merged_tags['relevance'] >= 0.5]

      # Group tags by movieID and merge them with movies
      grouped_tags = strong_tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
      movie_metadata = pd.merge(self.movies_df, grouped_tags, on='movieId', how='left')

      # Movie-metadata corpus
      movie_metadata['tag'] = movie_metadata['tag'].fillna('')
      metadata_corpus = movie_metadata['genres_str'] + ' ' + movie_metadata['tag']
    else:
      metadata_corpus = self.movies_df['genres_str']

    # Convert movie metadata into TF-IDF feature matrix
    self.tfidf_matrix = self.vectorizer.fit_transform(metadata_corpus).tocsr()

    # Pair movieId with index
    self.movie_idx = {movie_id: idx for idx, movie_id in enumerate(self.movies_df['movieId'])}

    # Compute TF-IDF vector magnitudes for cosine similarity
    self.tfidf_norms = np.asarray(np.sqrt(self.tfidf_matrix.power(2).sum(axis=1))).flatten()
    self.tfidf_norms[self.tfidf_norms == 0] = 1.0

    # Get unique users and pair with index
    unique_users = train_df['userId'].unique()
    self.user_idx_map = {u: i for i, u in enumerate(unique_users)}

    # Extract user and movieId values
    user_rows = train_df['userId'].map(self.user_idx_map).values
    movie_cols = train_df['movieId'].map(self.movie_idx).values

    # Mean-centre ratings by subtracting the global average
    centered_ratings = train_df['rating'].values - self.global_mean

    # Create user-item interaction matrix
    interaction_matrix = csr_matrix(
      (centered_ratings, (user_rows, movie_cols)),
      shape=(len(unique_users), self.tfidf_matrix.shape[0])
    )

    # Build user preference profiles by aggregating weighted movie TF-IDF features 
    self.user_profiles_matrix = interaction_matrix.dot(self.tfidf_matrix)
    self.user_profiles_norms = np.asarray(np.sqrt(self.user_profiles_matrix.power(2).sum(axis=1))).flatten()
    self.user_profiles_norms[self.user_profiles_norms == 0] = 1.0

    # Compute rating standard deviation for scaling similarity scores
    self.rating_std = float(train_df['rating'].std())

  def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:

    # Convert userId and movieId to 1D arrays
    user_ids = np.atleast_1d(user_ids)
    movie_ids = np.atleast_1d(movie_ids)

    # Fill preds with the global mean
    preds = np.full(len(user_ids), self.global_mean)

    # Map userId and movieId to internal matrix indices
    u_rows = pd.Series(user_ids).map(self.user_idx_map).fillna(-1).astype(int).values
    m_cols = pd.Series(movie_ids).map(self.movie_idx).fillna(-1).astype(int).values

    # Keep only valid user and movie mappings
    valid = (u_rows >= 0) & (m_cols >= 0)
    if not np.any(valid):
      return preds

    # Extract valid user and movie indices
    valid_u = u_rows[valid]
    valid_m = m_cols[valid]

    # Retrieve user preference vectors and candidate movie feature vectors
    u_vectors = self.user_profiles_matrix[valid_u]
    m_vectors = self.tfidf_matrix[valid_m]

    # Compute cosine similarity between user preference profile and candidate movie profile
    numerators = np.asarray(u_vectors.multiply(m_vectors).sum(axis=1)).flatten()
    denominators = self.user_profiles_norms[valid_u] * self.tfidf_norms[valid_m]

    # Clip cosine similarity scores within range [-1.0, 1.0]
    scores = np.clip(numerators / denominators, -1.0, 1.0)

    # Convert similarity scores into predicted ratings and clip to valid rating range
    preds[valid] = np.clip(self.global_mean + scores * self.rating_std, 0.5, 5.0)
    return preds

  def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:

    # Return empty if user not found
    if user_id not in self.user_idx_map:
      return pd.DataFrame(columns=['title', 'predicted_rating'])

    # Get user vector 
    u_row = self.user_idx_map[user_id]
    u_vector = self.user_profiles_matrix.getrow(u_row)

    # Compute cosine similarity scores for all candidate movies
    numerators = self.tfidf_matrix.dot(u_vector.T).toarray().flatten()
    denominators = self.user_profiles_norms[u_row] * self.tfidf_norms
    scores = np.clip(numerators / denominators, -1.0, 1.0)
    scaled_scores = np.clip(self.global_mean + scores * self.rating_std, 0.5, 5.0)

    # Construct recommendations df
    recommendations = pd.DataFrame({
      'movieId': self.movies_df['movieId'].values,
      'predicted_rating': scaled_scores
    })

    # Filter for unrated movies
    rated_movies = self.train_df[self.train_df['userId'] == user_id]['movieId']
    unrated = recommendations[~recommendations['movieId'].isin(rated_movies)]

    # Get top n recommendations
    top_df = unrated.nlargest(n, 'predicted_rating')
    result = top_df.merge(self.movies_df[['movieId', 'title']], on='movieId')[['title', 'predicted_rating']]
    result['predicted_rating'] = result['predicted_rating'].round(3)

    return result.reset_index(drop=True)