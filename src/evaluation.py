# Load dependencies

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from typing import Dict

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

  def full_evaluation(self, model, n: int = 10, threshold: float = 3.5) -> Dict[str, float]:

    # True ratings
    actuals = self.test_df['rating'].values

    # Predicted ratings
    preds = self._batched_predict(model, self.test_df['userId'].values, self.test_df['movieId'].values)

    results = {
      'RMSE': round(float(root_mean_squared_error(actuals, preds)), 3),
      'MAE': round(float(mean_absolute_error(actuals, preds)), 3)
    }

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