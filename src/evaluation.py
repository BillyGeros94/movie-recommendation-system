# Load dependencies

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from typing import Dict, Optional

class RecommenderEvaluator():

  def __init__(self, test_df: pd.DataFrame):

    # Store the test dataset
    self.test_df = test_df

  def _batched_predict(self, model, user_ids: np.ndarray, movie_ids: np.ndarray, chunk_size: int = 100000) -> np.ndarray:

    # Total number of user-movie pairs
    total_samples = len(user_ids)

    # Allocate memory for predictions
    all_preds = np.empty(total_samples, dtype=float)

     # Process predictions in chunks
    for start_idx in range(0, total_samples, chunk_size):

      # Set chunk boundaries
      end_idx = min(start_idx + chunk_size, total_samples)

      # Extract userId and movieId for this chunk
      u_chunk = user_ids[start_idx:end_idx]
      m_chunk = movie_ids[start_idx:end_idx]

      # Generate predictions for the chunk
      all_preds[start_idx:end_idx] = model.predict(u_chunk, m_chunk)

    return all_preds

  def full_evaluation(self, model, n: int = 10, threshold: float = 3.5) -> Dict[str, Optional[float]]:
    """Rating accuracy, plus re-ranking quality within each user's held-out movies.

    The ranking metrics here only reorder movies the user is already known to have rated, so
    even a random order scores well. Use catalogue_evaluation() to measure recommendation.
    """

    # True ratings
    actuals = self.test_df['rating'].values

    # Predicted ratings
    preds = self._batched_predict(model, self.test_df['userId'].values, self.test_df['movieId'].values)

    # Models that only rank have no meaningful rating error
    if model.rating_model:
      results = {
        'RMSE': round(float(root_mean_squared_error(actuals, preds)), 3),
        'MAE': round(float(mean_absolute_error(actuals, preds)), 3)
      }
    else:
      results = {'RMSE': None, 'MAE': None}

    # Create a copy of the test data
    test = self.test_df.copy()

    test['pred'] = preds

    # Rank predicted movies for each user and keep only the Top-N recommendations
    test['rank'] = test.groupby('userId')['pred'].rank(method='first', ascending=False)
    top_n = test[test['rank'] <= n].copy()

    # Recommendation is relevant if true rating >= threshold
    hits = (top_n['rating'] >= threshold).astype(int)

    # Average proportion of relevant recommendations
    precision = hits.groupby(top_n['userId']).mean().mean()

    # Count the number of relevant movies for each user
    relevant_counts = test[test['rating'] >= threshold].groupby('userId').size()

    # Compute recall for each user and average
    recall = (hits.groupby(top_n['userId']).sum() / relevant_counts).fillna(0).mean()

    # Compute Discounted Cumulative Gain (DCG)
    top_n['dcg'] = top_n['rating'] / np.log2(top_n['rank'] + 1)
    dcg_sum = top_n.groupby('userId')['dcg'].sum()

    # Create the ideal ranking using true ratings
    ideal_sorted = test.sort_values(['userId', 'rating'], ascending=[True, False])

    # Assign ideal ranking positions
    ideal_sorted['rank'] = ideal_sorted.groupby('userId')['rating'].cumcount() + 1

    # Keep the ideal Top-N items
    ideal_top_n = ideal_sorted[ideal_sorted['rank'] <= n].copy()

     # Compute Ideal DCG (IDCG)
    ideal_top_n['idcg'] = ideal_top_n['rating'] / np.log2(ideal_top_n['rank'] + 1)
    idcg_sum = ideal_top_n.groupby('userId')['idcg'].sum()

    # Normalised DCG averaged across users
    ndcg = (dcg_sum / idcg_sum).fillna(0).mean()

    results.update ({
      f'Precision@{n}': round(float(precision), 3),
      f'Recall@{n}': round(float(recall), 3),
      f'nDCG@{n}': round(float(ndcg), 3)
    })

    return results

  def catalogue_evaluation(self, model, train_df: pd.DataFrame, n: int = 10, threshold: float = 3.5,
                           users: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Recommendation quality: rank every movie a user has not rated in training, keep the top N.

    A hit is a recommended movie the user later rated at or above the threshold. Unrated does not
    mean disliked, so absolute values are lower bounds. Compare models against each other.
    Coverage is the share of the catalogue recommended to anyone. Popularity is the mean
    percentile of recommended movies by training rating count, where 100 is the most rated.
    """

    if users is None:
      users = self.test_df['userId'].unique()

    # Group training and relevant test movies by user, once
    seen_by_user = {u: g.values for u, g in train_df.loc[train_df['userId'].isin(users), 'movieId'].groupby(train_df['userId'])}
    relevant = self.test_df[(self.test_df['rating'] >= threshold) & self.test_df['userId'].isin(users)]
    relevant_by_user = {u: set(g.tolist()) for u, g in relevant['movieId'].groupby(relevant['userId'])}

    # Popularity percentile of every movie in the catalogue
    popularity = train_df['movieId'].value_counts().rank(pct=True) * 100

    discounts = 1.0 / np.log2(np.arange(2, n + 2))
    precisions, recalls, ndcgs = [], [], []
    recommended = []

    for user_id in users:
      user_id = int(user_id)
      movie_ids, scores = model.score_catalogue(user_id)

      # Top N among movies unseen in training
      scores = np.where(np.isin(movie_ids, seen_by_user.get(user_id, [])), -np.inf, scores)
      top = np.argpartition(-scores, n)[:n]
      top = top[np.argsort(-scores[top], kind='stable')]
      top_ids = movie_ids[top]
      recommended.append(top_ids)

      # Users with nothing relevant cannot be scored for recall or nDCG
      liked = relevant_by_user.get(user_id, set())
      hits = np.array([m in liked for m in top_ids.tolist()], dtype=float)
      precisions.append(hits.mean())
      if liked:
        recalls.append(hits.sum() / len(liked))
        ndcgs.append((hits * discounts).sum() / discounts[:min(len(liked), n)].sum())

    recommended = np.concatenate(recommended)
    catalogue_size = train_df['movieId'].nunique()

    return {
      f'Catalogue P@{n}': round(float(np.mean(precisions)), 4),
      f'Catalogue R@{n}': round(float(np.mean(recalls)), 4),
      f'Catalogue nDCG@{n}': round(float(np.mean(ndcgs)), 4),
      'Coverage': round(float(len(np.unique(recommended)) / catalogue_size), 4),
      'Popularity percentile': round(float(popularity.reindex(recommended).mean()), 2)
    }
