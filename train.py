"""Train every model, evaluate it, and write the artifacts the API and demo load.

  python train.py                        # everything, on a fixed sample of test users
  python train.py --models uv item       # selected models only
  python train.py --eval-users 0         # evaluate on every test user (user-based CF is slow)

Results are merged into results/metrics.json, so models can be run separately.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import config
from baselines import BiasBaseline, ItemMeanBaseline, PopularityBaseline, RandomBaseline
from content_model import ContentBasedRecommender
from data import download_dataset, load_genome, load_movies, prepare_split, sample_users
from evaluation import RecommenderEvaluator
from fold_in import fold_in_user, score_new_user
from item_model import ItemBasedCF
from user_model import UserBasedCF
from uv_model import UVDecomposition

API_MODELS = ROOT / 'api' / 'models'
DEMO_MODEL = ROOT / 'streamlit' / 'uv_model_deploy.joblib'
METRICS_FILE = ROOT / 'results' / 'metrics.json'

MODEL_NAMES = ['random', 'popularity', 'item_mean', 'bias', 'content', 'item', 'uv', 'user']

def save_metrics(section: str, key: str, values: dict):

  # Merge into the results file, so separate runs build one report
  metrics = json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}
  metrics.setdefault(section, {})[key] = values
  METRICS_FILE.parent.mkdir(exist_ok=True)
  METRICS_FILE.write_text(json.dumps(metrics, indent=2))

def fit_model(name, train, movies, dataset_dir):

  # Build and train one model
  if name == 'random':
    model = RandomBaseline()
  elif name == 'popularity':
    model = PopularityBaseline()
  elif name == 'item_mean':
    model = ItemMeanBaseline()
  elif name == 'bias':
    model = BiasBaseline()
  elif name == 'content':
    model = ContentBasedRecommender()
    tags, scores = load_genome(dataset_dir)
    model.fit(train, movies, tags, scores)
    return model
  elif name == 'item':
    model = ItemBasedCF(**config.ITEM_CF)
  elif name == 'user':
    model = UserBasedCF(**config.USER_CF)
  elif name == 'uv':
    model = UVDecomposition(**config.UV_INIT)
    model.fit(train, movies, **config.UV_FIT)
    return model
  model.fit(train, movies)
  return model

def save_artifacts(name, model):

  # Slim models hold no user data, so they are small enough to keep in the repository
  API_MODELS.mkdir(parents=True, exist_ok=True)
  if name in ('content', 'item', 'uv'):
    joblib.dump(model.for_serving(), API_MODELS / f'{name}.joblib', compress=3)

  # The demo artifact is plain arrays, so it needs neither this package's classes nor scipy
  if name == 'uv':
    joblib.dump({
      'Q': model.Q,
      'b_i': model.b_i,
      'global_mean': model.global_mean,
      'fold_in_reg': model.fold_in_reg,
      'movie_ids': model.movie_ids,
      'titles': np.array([model.titles[int(m)] for m in model.movie_ids], dtype=object)
    }, DEMO_MODEL, compress=3)

def fold_in_evaluation(model, train, test, users, n_ratings=5, n=10, threshold=3.5):

  # Replays the demo: each user arrives with only their last few training ratings
  movie_idx = model.movie_idx
  squared_errors, item_only_errors, precisions, popularity_precisions = [], [], [], []
  popularity = np.bincount(train['movieId'].map(movie_idx).values, minlength=len(model.b_i)).astype(float)
  train_groups = {u: g for u, g in train[train['userId'].isin(users)].groupby('userId')}
  test_groups = {u: g for u, g in test[test['userId'].isin(users)].groupby('userId')}

  for user_id in users:
    recent = train_groups[user_id].tail(n_ratings)
    held_out = test_groups[user_id]
    rated = recent['movieId'].map(movie_idx).values
    targets = held_out['movieId'].map(movie_idx).values

    b_u, p_u = fold_in_user(model.Q, model.b_i, model.global_mean, rated, recent['rating'].values, reg=model.fold_in_reg)
    scores = score_new_user(model.Q, model.b_i, model.global_mean, b_u, p_u)
    squared_errors.append((np.clip(scores[targets], 0.5, 5.0) - held_out['rating'].values) ** 2)

    # The same prediction with nothing learned about the user
    item_only = np.clip(model.global_mean + model.b_i[targets], 0.5, 5.0)
    item_only_errors.append((item_only - held_out['rating'].values) ** 2)

    # A new user has only rated these few movies, so everything else is a candidate
    scores[rated] = -np.inf
    top = np.argpartition(-scores, n)[:n]
    liked = set(targets[held_out['rating'].values >= threshold].tolist())
    precisions.append(len(liked.intersection(top.tolist())) / n)

    # Popularity gets the same candidates, so the two precisions compare directly
    popular = popularity.copy()
    popular[rated] = -np.inf
    popularity_precisions.append(len(liked.intersection(np.argpartition(-popular, n)[:n].tolist())) / n)

  return {
    'n_ratings': n_ratings,
    'RMSE': round(float(np.sqrt(np.concatenate(squared_errors).mean())), 3),
    'Item-only RMSE': round(float(np.sqrt(np.concatenate(item_only_errors).mean())), 3),
    f'Catalogue P@{n}': round(float(np.mean(precisions)), 4),
    f'Popularity P@{n}': round(float(np.mean(popularity_precisions)), 4)
  }

def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--data-dir', default=str(ROOT / 'data'))
  parser.add_argument('--models', nargs='+', default=MODEL_NAMES, choices=MODEL_NAMES)
  parser.add_argument('--eval-users', type=int, default=5000, help='test users for rating metrics, 0 for all')
  parser.add_argument('--catalogue-users', type=int, default=1000, help='users for full-catalogue metrics, 0 for all')
  parser.add_argument('--no-artifacts', action='store_true', help='evaluate without overwriting saved models')
  args = parser.parse_args()

  dataset_dir = download_dataset(args.data_dir)
  train, test = prepare_split(args.data_dir)
  movies = load_movies(dataset_dir)
  print(f'Train: {len(train):,} ratings. Test: {len(test):,} ratings.', flush=True)

  # Every model is scored on the same users
  eval_users = sample_users(test, args.eval_users)
  test_sample = test[test['userId'].isin(eval_users)]
  evaluator = RecommenderEvaluator(test_sample)
  catalogue_users = sample_users(test_sample, args.catalogue_users)
  print(f'Evaluating on {len(eval_users):,} users, {len(test_sample):,} test ratings.', flush=True)

  for name in args.models:
    started = time.time()
    model = fit_model(name, train, movies, dataset_dir)
    print(f'{name}: trained in {time.time() - started:.0f}s', flush=True)

    if not args.no_artifacts:
      save_artifacts(name, model)

    if hasattr(model, 'reset_fallback_stats'):
      model.reset_fallback_stats()

    results = evaluator.full_evaluation(model)
    if hasattr(model, 'fallback_rate'):
      results['Pairs without neighbours'] = round(model.fallback_rate, 4)

    results.update(evaluator.catalogue_evaluation(model, train, users=catalogue_users))
    results['seconds'] = round(time.time() - started)
    print(name, results, flush=True)
    save_metrics('models', name, results)
    save_metrics('run', 'sample', {'eval_users': len(eval_users), 'test_ratings': len(test_sample),
                                   'catalogue_users': len(catalogue_users)})

    if name == 'uv':
      for n_ratings in (5, 20):
        values = fold_in_evaluation(model, train, test_sample, catalogue_users, n_ratings=n_ratings)
        print('fold-in', values, flush=True)
        save_metrics('fold_in', str(n_ratings), values)

    del model
    gc.collect()

if __name__ == '__main__':
  main()
