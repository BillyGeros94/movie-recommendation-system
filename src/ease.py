# Load Dependencies

import numpy as np
from scipy.sparse import csr_matrix

def solve_ease(user_rows: np.ndarray, item_columns: np.ndarray, n_items: int, reg: float = 500.0) -> np.ndarray:
  """Closed-form EASE weights from who rated what, ignoring the rating values.

  Each column of the result is a ridge regression predicting whether a user rated that movie
  from the other movies they rated. One matrix inverse solves all of them at once.
  Row = a movie the user rated, column = the movie being scored.
  """

  values = np.ones(len(user_rows), dtype=np.float64)
  matrix = csr_matrix((values, (user_rows, item_columns)), shape=(int(user_rows.max()) + 1, n_items))

  gram = (matrix.T @ matrix).toarray()
  gram[np.diag_indices(n_items)] += reg

  inverse = np.linalg.inv(gram)
  weights = -inverse / np.diag(inverse)
  np.fill_diagonal(weights, 0)

  return weights.astype(np.float32)
