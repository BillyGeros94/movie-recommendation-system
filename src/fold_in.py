# Load Dependencies

from typing import Optional, Tuple

import numpy as np

def fold_in_user(Q: np.ndarray, b_i: np.ndarray, global_mean: float, item_indices: np.ndarray,
                 ratings: np.ndarray, reg: float = 2.0, bias_damping: float = 5.0) -> Tuple[float, np.ndarray]:
  """Estimate a new user's bias and latent vector from their ratings, with Q held fixed.

  Solves the same regularised least squares problem as training, for one user. Ratings below
  what the model expects give a negative residual, so disliked movies push the vector away.
  """

  item_indices = np.asarray(item_indices)
  ratings = np.asarray(ratings, dtype=np.float64)
  factors = Q[item_indices].astype(np.float64)

  # Damped user bias, so a handful of ratings cannot all be explained away as harsh or generous rating
  expected = global_mean + b_i[item_indices]
  b_u = float((ratings - expected).sum() / (len(ratings) + bias_damping))

  # Ridge solution: (Q'Q + reg I) p = Q'(r - baseline)
  residuals = ratings - expected - b_u
  gram = factors.T @ factors + reg * np.eye(Q.shape[1])
  p_u = np.linalg.solve(gram, factors.T @ residuals)

  return b_u, p_u

def score_new_user(Q: np.ndarray, b_i: np.ndarray, global_mean: float, b_u: float, p_u: np.ndarray) -> np.ndarray:

  # Unclipped predicted rating for every movie
  return global_mean + b_u + b_i + Q @ p_u

def rank_new_user(Q: np.ndarray, b_i: np.ndarray, global_mean: float, item_indices: np.ndarray,
                  ratings: np.ndarray, candidates: np.ndarray, reg: float = 2.0) -> np.ndarray:
  """Score movies for choosing a new user's list. Use these to rank, not as predicted ratings.

  Differs from fold_in_user in two ways. No user offset is removed first, because people tend to
  rate only what they love, and removing it would leave little taste signal. Only candidates can
  be chosen, because thinly rated movies have unreliable factors and would otherwise win on noise.
  Everything else scores -inf, including the movies the user rated.
  """

  item_indices = np.asarray(item_indices)
  ratings = np.asarray(ratings, dtype=np.float64)
  factors = Q[item_indices].astype(np.float64)

  residuals = ratings - global_mean - b_i[item_indices]
  gram = factors.T @ factors + reg * np.eye(Q.shape[1])
  taste = np.linalg.solve(gram, factors.T @ residuals)

  scores = np.full(len(b_i), -np.inf)
  scores[candidates] = b_i[candidates] + Q[candidates] @ taste
  scores[item_indices] = -np.inf

  return scores

def ease_shortlist(ease_weights: np.ndarray, candidates: np.ndarray, item_indices: np.ndarray, ratings: np.ndarray,
                   n_items: int, size: int = 30, liked_rating: float = 3.5) -> Optional[np.ndarray]:
  """Pick the movies a user is most likely to watch next, from the movies they liked.

  EASE only knows rated or not rated, so a low rating would count as evidence for similar movies.
  Movies below liked_rating are therefore treated as not seen. Returns None with no usable evidence.
  """

  item_indices = np.asarray(item_indices)
  ratings = np.asarray(ratings, dtype=np.float64)

  # Position of each movie within the candidates, -1 for the rest
  slots = np.full(n_items, -1)
  slots[candidates] = np.arange(len(candidates))

  rated_slots = slots[item_indices]
  liked_slots = rated_slots[(ratings >= liked_rating) & (rated_slots >= 0)]
  if len(liked_slots) == 0:
    return None

  scores = ease_weights[liked_slots].sum(axis=0, dtype=np.float64)
  scores[rated_slots[rated_slots >= 0]] = -np.inf

  top = np.argsort(-scores, kind='stable')[:size]
  return candidates[top[np.isfinite(scores[top])]]

def rank_hybrid(Q: np.ndarray, b_i: np.ndarray, global_mean: float, item_indices: np.ndarray, ratings: np.ndarray,
                candidates: np.ndarray, ease_weights: Optional[np.ndarray], reg: float = 2.0, shortlist_size: int = 30,
                liked_rating: float = 3.5) -> np.ndarray:
  """EASE chooses a shortlist of likely watches, then the UV ranking orders it by taste.

  Falls back to the UV ranking alone when there is no shortlist, for example if every rating is low.
  """

  ranking = rank_new_user(Q, b_i, global_mean, item_indices, ratings, candidates, reg=reg)
  if ease_weights is None:
    return ranking

  shortlist = ease_shortlist(ease_weights, candidates, item_indices, ratings, len(b_i), shortlist_size, liked_rating)
  if shortlist is None:
    return ranking

  scores = np.full(len(b_i), -np.inf)
  scores[shortlist] = ranking[shortlist]
  return scores
