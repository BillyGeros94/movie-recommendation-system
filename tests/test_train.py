import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from train import fold_in_evaluation
from uv_model import UVDecomposition

def test_fold_in_evaluation_reports_comparators(ratings, movies):
  train = ratings.groupby('userId').head(28)
  test = ratings.drop(train.index)
  model = UVDecomposition(n_factors=4, std=0.1)
  model.fit(train, movies, n_epochs=50, lr=0.02, batch_size=500)

  result = fold_in_evaluation(model, train, test, train['userId'].unique(), n_ratings=5)

  # The two taste groups are visible from five ratings, which item averages cannot use
  assert result['RMSE'] < result['Item-only RMSE']
  assert 0 <= result['Popularity P@10'] <= 1
