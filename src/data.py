# Load Dependencies

import io
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests

DATASET_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'

def download_dataset(data_dir: Path) -> Path:

  # Download and extract once, then reuse the local copy
  data_dir = Path(data_dir)
  dataset_dir = data_dir / 'ml-latest'
  if not (dataset_dir / 'ratings.csv').exists():
    response = requests.get(DATASET_URL, timeout=600)
    response.raise_for_status()
    zipfile.ZipFile(io.BytesIO(response.content)).extractall(data_dir)
  return dataset_dir

def load_ratings(dataset_dir: Path) -> pd.DataFrame:

  # Compact dtypes roughly halve memory on 33M rows
  return pd.read_csv(
    Path(dataset_dir) / 'ratings.csv',
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32', 'timestamp': 'int64'}
  )

def load_movies(dataset_dir: Path) -> pd.DataFrame:
  return pd.read_csv(Path(dataset_dir) / 'movies.csv')

def load_genome(dataset_dir: Path, min_relevance: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:

  # Stream the 18M genome scores and keep only the strong ones
  tags = pd.read_csv(Path(dataset_dir) / 'genome-tags.csv')
  chunks = pd.read_csv(Path(dataset_dir) / 'genome-scores.csv', chunksize=2_000_000)
  scores = pd.concat(chunk[chunk['relevance'] >= min_relevance] for chunk in chunks)
  return tags, scores

def co_filter(ratings: pd.DataFrame, min_ratings: int = 100) -> pd.DataFrame:

  # Filter movies and users alternately until the dataset stabilises
  filtered = ratings
  while True:
    initial_rows = len(filtered)

    movie_counts = filtered['movieId'].value_counts()
    filtered = filtered[filtered['movieId'].isin(movie_counts[movie_counts >= min_ratings].index)]

    user_counts = filtered['userId'].value_counts()
    filtered = filtered[filtered['userId'].isin(user_counts[user_counts >= min_ratings].index)]

    if len(filtered) == initial_rows:
      return filtered

def temporal_split(ratings: pd.DataFrame, train_fraction: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:

  # Per-user chronological split: each user's earliest ratings go to train
  ordered = ratings.sort_values(['userId', 'timestamp']).reset_index(drop=True)
  user_totals = ordered.groupby('userId')['userId'].transform('count')
  user_ranks = ordered.groupby('userId').cumcount() + 1
  train_mask = user_ranks <= user_totals * train_fraction
  return ordered[train_mask], ordered[~train_mask]

def prepare_split(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:

  # Filter and split once, then reuse the cached result
  cache = Path(data_dir) / 'split.pkl'
  if cache.exists():
    return pd.read_pickle(cache)

  ratings = co_filter(load_ratings(download_dataset(data_dir)))
  ratings['rating'] = ratings['rating'].astype('float64')
  train, test = temporal_split(ratings)
  pd.to_pickle((train, test), cache)
  return train, test

def sample_users(ratings: pd.DataFrame, n_users: int, seed: int = 42) -> np.ndarray:

  # Reproducible user sample. n_users of 0 means every user
  users = np.sort(ratings['userId'].unique())
  if n_users <= 0 or n_users >= len(users):
    return users
  return np.sort(np.random.default_rng(seed).choice(users, n_users, replace=False))
