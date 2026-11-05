# Load Dependencies

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from base_model import BaseRecommender

def split_genres(genres) -> List[str]:

  # Genres arrive as 'A|B' from the CSV, or as a list once parsed
  if isinstance(genres, (list, tuple, np.ndarray)):
    values = list(genres)
  elif pd.isna(genres):
    values = []
  else:
    values = str(genres).split('|')
  return [g for g in values if g and g != '(no genres listed)']

class ContentBasedRecommender(BaseRecommender):

  training_only = ('train_df', 'interactions', 'user_means', 'user_idx')

  # Object initialization
  def __init__(self):
    self.vectorizer = TfidfVectorizer(max_df=0.8, stop_words='english')
    self.interactions = None
    self.user_means = None
    self.user_idx = {}
    self.tfidf_matrix = None
    self.movie_idx = {}
    self.train_df = None
    self.genres = []
    self.global_mean = 3.5
    self.rating_std = 1.0

  def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame, tags_df: pd.DataFrame = None, scores_df: pd.DataFrame = None):

    # Compute global mean rating and spread
    self.global_mean = float(train_df['rating'].mean())
    self.rating_std = float(train_df['rating'].std())

    # The catalogue is the set of movies with training ratings
    catalogue = movies_df[movies_df['movieId'].isin(train_df['movieId'].unique())].reset_index(drop=True)
    self._store_catalogue(train_df, movies_df, catalogue['movieId'].values)

    # Prepare genre metadata
    genre_lists = catalogue['genres'].apply(split_genres)
    self.genres = sorted({genre for genres in genre_lists for genre in genres})
    metadata_corpus = genre_lists.apply(' '.join)

    if tags_df is not None and scores_df is not None:
      # Merge tag metadata with relevance scores and keep strong tags
      strong_scores = scores_df[scores_df['relevance'] >= 0.5]
      strong_tags = pd.merge(strong_scores, tags_df, on='tagId')

      # Group tags by movieId and append them to the genres
      grouped_tags = strong_tags.groupby('movieId')['tag'].apply(' '.join)
      metadata_corpus = metadata_corpus + ' ' + catalogue['movieId'].map(grouped_tags).fillna('')

    # Convert movie metadata into TF-IDF feature matrix, with unit-length rows
    self.tfidf_matrix = self.vectorizer.fit_transform(metadata_corpus).tocsr()

    # Create mappings between IDs and matrix indices
    self.movie_idx = {m: i for i, m in enumerate(self.movie_ids.tolist())}
    users = train_df['userId'].unique()
    self.user_idx = {u: i for i, u in enumerate(users.tolist())}

    # Extract user and movie indices
    user_rows = train_df['userId'].map(self.user_idx).values
    movie_cols = train_df['movieId'].map(self.movie_idx).values

    # Centre on each user's own mean, so a movie counts as liked relative to how that user rates
    self.user_means = train_df.groupby('userId')['rating'].mean().reindex(users).values.astype(np.float64)
    centered_ratings = train_df['rating'].values - self.user_means[user_rows]

    # Store weights rather than profiles: built on demand, because 81K stored profiles would need over 1GB
    self.interactions = csr_matrix(
      (centered_ratings.astype(np.float32), (user_rows, movie_cols)),
      shape=(len(users), self.tfidf_matrix.shape[0])
    )

  def _similarities(self, profile: csr_matrix) -> np.ndarray:

    # Cosine similarity between a profile and every movie
    norm = np.sqrt(profile.multiply(profile).sum())
    if norm == 0:
      return np.zeros(self.tfidf_matrix.shape[0])
    return self.tfidf_matrix.dot(profile.T).toarray().flatten() / norm

  def _user_scores(self, u_row: int) -> np.ndarray:

    # Weighted sum of rated movie vectors: liked movies pull the profile in, disliked movies push it away
    profile = self.interactions.getrow(u_row).dot(self.tfidf_matrix)
    return self.user_means[u_row] + self._similarities(profile) * self.rating_std

  def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:

    # Convert userId and movieId to 1D arrays
    user_ids = np.atleast_1d(user_ids)
    movie_ids = np.atleast_1d(movie_ids)

    # Fill preds with the global mean
    preds = np.full(len(user_ids), self.global_mean)

    # Map userId and movieId to internal matrix indices
    u_rows = pd.Series(user_ids).map(self.user_idx).fillna(-1).astype(int).values
    m_cols = pd.Series(movie_ids).map(self.movie_idx).fillna(-1).astype(int).values

    # An unknown movie falls back to the user's mean
    known_users = u_rows >= 0
    preds[known_users] = self.user_means[u_rows[known_users]]

    # Score each user's candidates together
    known = np.flatnonzero(known_users & (m_cols >= 0))
    for u_row, positions in pd.Series(known).groupby(u_rows[known]):
      positions = positions.values
      preds[positions] = self._user_scores(u_row)[m_cols[positions]]

    return np.clip(preds, 0.5, 5.0)

  def score_catalogue(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
    return self.movie_ids, self._user_scores(self.user_idx[user_id])

  def recommend_from_text(self, text: str, n: int = 10) -> pd.DataFrame:

    # Serve a free-text profile without changing model state
    query = self.vectorizer.transform([text])
    if query.nnz == 0:
      return pd.DataFrame(columns=['movieId', 'title', 'predicted_rating'])

    scores = self.global_mean + self._similarities(query) * self.rating_std
    return self._top_n(self.movie_ids, scores, set(), n)
