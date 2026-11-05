"""Choose hyperparameters on a validation split. The test set is never touched.

The last eighth of each user's training ratings becomes validation, which keeps the split
chronological. Grids run on a sample of users so they finish on modest hardware. Results go to
results/tuning.json, and the chosen values are copied into src/config.py by hand.

  python tune.py --models uv item user fold_in
  python tune.py --models uv_epochs item_full --tune-users 0
"""

import argparse
import gc
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import config
from data import download_dataset, load_movies, prepare_split, sample_users, temporal_split
from fold_in import fold_in_user, score_new_user
from item_model import ItemBasedCF
from user_model import UserBasedCF
from uv_model import UVDecomposition

TUNING_FILE = ROOT / 'results' / 'tuning.json'

UV_GRID = {'lr': [0.005, 0.01, 0.02], 'reg': [0.02, 0.05, 0.1]}
ITEM_GRID = {'n_neighbors': [50, 200, 400], 'sim_shrinkage': [25.0, 100.0], 'weight_shrinkage': [0.01, 0.03, 0.1, 1.0]}
USER_GRID = {'n_neighbors': [25, 50, 100], 'sim_shrinkage': [10.0, 50.0], 'weight_shrinkage': [0.01, 0.03, 0.1, 1.0]}
ITEM_FULL_SHRINKAGE = [5.0, 10.0, 25.0, 100.0, 400.0]
FOLD_IN_REGS = [0.5, 1.0, 2.0, 5.0, 10.0]

def save(section, values):
  results = json.loads(TUNING_FILE.read_text()) if TUNING_FILE.exists() else {}
  results[section] = values
  TUNING_FILE.parent.mkdir(exist_ok=True)
  TUNING_FILE.write_text(json.dumps(results, indent=2))

def rmse(model, ratings):
  errors = ratings['rating'].values - model.predict(ratings['userId'].values, ratings['movieId'].values)
  return round(float(np.sqrt(np.mean(errors ** 2))), 4)

def grid(options):
  keys = list(options)
  return [dict(zip(keys, values)) for values in itertools.product(*options.values())]

def tune_uv(train, val, movies, n_epochs):
  rows = []
  for params in grid(UV_GRID):
    model = UVDecomposition()
    model.fit(train, movies, n_epochs=n_epochs, val_df=val, **params)
    rows.append({**params, 'best_epoch': model.best_epoch, 'val_rmse': min(h['val_rmse'] for h in model.history)})
    print(rows[-1], flush=True)
    save('uv_grid', rows)
    del model
    gc.collect()

def tune_item(train, val, movies):
  rows = []
  for params in grid(ITEM_GRID):
    model = ItemBasedCF(**params)
    model.fit(train, movies)
    rows.append({**params, 'val_rmse': rmse(model, val), 'fallback': round(model.fallback_rate, 4)})
    print(rows[-1], flush=True)
    save('item_grid', rows)
    del model
    gc.collect()

def tune_user(train, val, movies, n_users):
  rows = []

  # Similarities are the slow part, so score a smaller sample of validation users
  val_sample = val[val['userId'].isin(sample_users(val, n_users))]
  for params in grid(USER_GRID):
    model = UserBasedCF(**params)
    model.fit(train, movies)
    rows.append({**params, 'val_rmse': rmse(model, val_sample), 'fallback': round(model.fallback_rate, 4)})
    print(rows[-1], flush=True)
    save('user_grid', rows)
    del model
    gc.collect()

def tune_uv_epochs(train, val, movies, n_epochs):

  # The stopping epoch depends on data size, so it is found on the full training share
  model = UVDecomposition(**config.UV_INIT)
  params = {k: v for k, v in config.UV_FIT.items() if k != 'n_epochs'}
  model.fit(train, movies, n_epochs=n_epochs, val_df=val, **params)
  save('uv_epochs', {'best_epoch': model.best_epoch, 'history': model.history})

def tune_item_full(train, val, movies, n_users):

  # Co-rating counts grow with the number of users, so shrinkage is checked on the full training share
  rows = []
  val_sample = val[val['userId'].isin(sample_users(val, n_users))]
  for sim_shrinkage in ITEM_FULL_SHRINKAGE:
    model = ItemBasedCF(**{**config.ITEM_CF, 'sim_shrinkage': sim_shrinkage})
    model.fit(train, movies)
    rows.append({'sim_shrinkage': sim_shrinkage, 'val_rmse': rmse(model, val_sample), 'fallback': round(model.fallback_rate, 4)})
    print(rows[-1], flush=True)
    save('item_full', rows)
    del model
    gc.collect()

def tune_fold_in(train, val, movies, n_epochs, uv_fit):

  # Replays the demo: rebuild each user from their last 5 ratings and predict their validation ratings
  model = UVDecomposition()
  model.fit(train, movies, n_epochs=n_epochs, **uv_fit)
  train_groups = dict(tuple(train.groupby('userId')))
  val_groups = dict(tuple(val.groupby('userId')))
  rows = []
  for reg in FOLD_IN_REGS:
    errors = []
    for user_id, held_out in val_groups.items():
      recent = train_groups[user_id].tail(5)
      rated = recent['movieId'].map(model.movie_idx).values
      targets = held_out['movieId'].map(model.movie_idx)
      known = targets.notna().values
      b_u, p_u = fold_in_user(model.Q, model.b_i, model.global_mean, rated, recent['rating'].values, reg=reg)
      scores = score_new_user(model.Q, model.b_i, model.global_mean, b_u, p_u)
      errors.append(np.clip(scores[targets[known].astype(int).values], 0.5, 5.0) - held_out['rating'].values[known])
    rows.append({'fold_in_reg': reg, 'val_rmse': round(float(np.sqrt(np.mean(np.concatenate(errors) ** 2))), 4)})
    print(rows[-1], flush=True)
  save('fold_in', rows)

def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--data-dir', default=str(ROOT / 'data'))
  parser.add_argument('--models', nargs='+', default=['uv', 'item', 'user', 'fold_in'], choices=['uv', 'item', 'user', 'fold_in', 'uv_epochs', 'item_full'])
  parser.add_argument('--tune-users', type=int, default=8000, help='users in the tuning sample, 0 for all')
  parser.add_argument('--val-users', type=int, default=1000, help='validation users scored for user-based CF and item_full')
  parser.add_argument('--n-epochs', type=int, default=40)
  args = parser.parse_args()

  dataset_dir = download_dataset(args.data_dir)
  train, _ = prepare_split(args.data_dir)
  movies = load_movies(dataset_dir)

  # 7/8 of the 80% training share leaves 70/10/20 overall
  train = train[train['userId'].isin(sample_users(train, args.tune_users))]
  fit_part, val = temporal_split(train, train_fraction=0.875)
  val = val[val['movieId'].isin(fit_part['movieId'].unique())]
  print(f'Tuning on {len(fit_part):,} ratings, validating on {len(val):,}.', flush=True)

  if 'uv' in args.models:
    tune_uv(fit_part, val, movies, args.n_epochs)
  if 'item' in args.models:
    tune_item(fit_part, val, movies)
  if 'user' in args.models:
    tune_user(fit_part, val, movies, args.val_users)
  if 'fold_in' in args.models:
    tune_fold_in(fit_part, val, movies, args.n_epochs, {'lr': config.UV_FIT['lr'], 'reg': config.UV_FIT['reg']})
  if 'uv_epochs' in args.models:
    tune_uv_epochs(fit_part, val, movies, args.n_epochs)
  if 'item_full' in args.models:
    tune_item_full(fit_part, val, movies, args.val_users)

if __name__ == '__main__':
  main()
