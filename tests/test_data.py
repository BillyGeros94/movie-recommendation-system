import pandas as pd

from data import co_filter, sample_users, temporal_split

def test_co_filter_repeats_until_stable():

  # Removing user 3 drops movie 20 below the threshold, which only a second pass catches
  ratings = pd.DataFrame(
    [(1, 10), (1, 20), (2, 10), (2, 20), (3, 20), (3, 30)] + [(4, 10), (4, 40), (5, 40), (5, 10)],
    columns=['userId', 'movieId']
  )
  filtered = co_filter(ratings, min_ratings=2)
  assert (filtered['movieId'].value_counts() >= 2).all()
  assert (filtered['userId'].value_counts() >= 2).all()

def test_temporal_split_keeps_each_users_earliest_ratings_in_train():
  ratings = pd.DataFrame({
    'userId': [1] * 10 + [2] * 5,
    'movieId': list(range(10)) + list(range(5)),
    'rating': [4.0] * 15,
    'timestamp': list(range(10, 0, -1)) + list(range(100, 105))
  })
  train, test = temporal_split(ratings, train_fraction=0.8)
  assert len(train) == 12 and len(test) == 3
  for user_id in (1, 2):
    assert train[train['userId'] == user_id]['timestamp'].max() < test[test['userId'] == user_id]['timestamp'].min()

def test_sample_users_is_reproducible():
  ratings = pd.DataFrame({'userId': range(100)})
  assert (sample_users(ratings, 10) == sample_users(ratings, 10)).all()
  assert len(sample_users(ratings, 0)) == 100
