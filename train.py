"""Train every model, evaluate it, and write the artifacts the API and demo load.

  python train.py                        # everything, on a fixed sample of test users
  python train.py --models uv item       # selected models only
  python train.py --eval-users 0         # evaluate on every test user (user-based CF is slow)
  python train.py --new-users-only       # new-user figures from the saved UV model, without training

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
from fold_in import ease_shortlist, fold_in_user, rank_hybrid, rank_new_user, score_new_user
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
    model.fit_ease(train)
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
      'candidates': model.candidates,
      'ease_weights': model.ease_weights,
      'shortlist_size': model.shortlist_size,
      'liked_rating': model.liked_rating,
      'movie_ids': model.movie_ids,
      'titles': np.array([model.titles[int(m)] for m in model.movie_ids], dtype=object)
    }, DEMO_MODEL, compress=3)

def fold_in_evaluation(model, train, test, users, n_ratings=5, n=10, threshold=3.5):

  # Replays the demo: each user arrives with only their last few training ratings
  movie_idx = model.movie_idx
  squared_errors, item_only_errors = [], []
  precisions = {'UV': [], 'Hybrid': [], 'Popularity': [], 'EASE': []}
  pick_ratings = {name: [] for name in precisions}
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

    # Choose the lists as the demo does. A new user has only rated these few movies
    values = recent['rating'].values
    popular = popularity.copy()
    popular[rated] = -np.inf

    # EASE alone: its own top picks, best first. With no liked movie it has no evidence, so popularity stands in
    shortlist = ease_shortlist(model.ease_weights, model.candidates, rated, values, len(model.b_i), size=n,
                               liked_rating=model.liked_rating)
    if shortlist is None:
      ease_alone = popular
    else:
      ease_alone = np.full(len(model.b_i), -np.inf)
      ease_alone[shortlist] = np.arange(len(shortlist), 0, -1)

    lists = {
      'UV': rank_new_user(model.Q, model.b_i, model.global_mean, rated, values, model.candidates, reg=model.fold_in_reg),
      'Hybrid': rank_hybrid(model.Q, model.b_i, model.global_mean, rated, values, model.candidates, model.ease_weights,
                            reg=model.fold_in_reg, shortlist_size=max(model.shortlist_size, n), liked_rating=model.liked_rating),
      'Popularity': popular,
      'EASE': ease_alone
    }

    # A pick counts when the user went on to rate it, and is a hit when they liked it
    actual = dict(zip(targets.tolist(), held_out['rating'].values.tolist()))
    for name, scores_for_list in lists.items():
      top = np.argpartition(-scores_for_list, n)[:n]
      known = [actual[i] for i in top.tolist() if i in actual]
      precisions[name].append(sum(rating >= threshold for rating in known) / n)
      pick_ratings[name].extend(known)

  results = {
    'n_ratings': n_ratings,
    'RMSE': round(float(np.sqrt(np.concatenate(squared_errors).mean())), 3),
    'Item-only RMSE': round(float(np.sqrt(np.concatenate(item_only_errors).mean())), 3)
  }
  for name in precisions:
    results[f'{name} P@{n}'] = round(float(np.mean(precisions[name])), 4)
    results[f'{name} pick rating'] = round(float(np.mean(pick_ratings[name])), 2) if pick_ratings[name] else None

  return results

def new_user_evaluation(model, train, test, eval_users, catalogue_users):

  # The new-user settings were compared on catalogue_users, so the fairer figure comes from 1,000 test users
  # outside the evaluation sample, which no model or setting was chosen on
  test_sample = test[test['userId'].isin(eval_users)]
  other_users = sample_users(test[~test['userId'].isin(eval_users)], len(catalogue_users))
  results = {}
  for section, users in (('fold_in', catalogue_users), ('fold_in_other_users', other_users)):
    for n_ratings in (5, 20):
      results[(section, str(n_ratings))] = fold_in_evaluation(model, train, test_sample if section == 'fold_in' else test,
                                                              users, n_ratings=n_ratings)
  return results

def check_matches_saved(results):

  # Recomputing from the saved model must reproduce the saved figures exactly. If it does not, the data, the split
  # or the model differ from the run behind results/metrics.json, so nothing is written
  saved = json.loads(METRICS_FILE.read_text()).get('fold_in', {}) if METRICS_FILE.exists() else {}
  for key, old in saved.items():
    new = results[('fold_in', key)]
    changed = {name: (value, new.get(name)) for name, value in old.items() if new.get(name) != value}
    if changed:
      raise SystemExit(f'fold_in {key} does not match results/metrics.json, so nothing was written: {changed}')

def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--data-dir', default=str(ROOT / 'data'))
  parser.add_argument('--models', nargs='+', default=MODEL_NAMES, choices=MODEL_NAMES)
  parser.add_argument('--eval-users', type=int, default=5000, help='test users for rating metrics, 0 for all')
  parser.add_argument('--catalogue-users', type=int, default=1000, help='users for full-catalogue metrics, 0 for all')
  parser.add_argument('--no-artifacts', action='store_true', help='evaluate without overwriting saved models')
  parser.add_argument('--new-users-only', action='store_true',
                      help='new-user figures from the saved UV model, without training or touching any model file')
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

  if args.new_users_only:
    results = new_user_evaluation(joblib.load(API_MODELS / 'uv.joblib'), train, test, eval_users, catalogue_users)
    check_matches_saved(results)
    for (section, key), values in results.items():
      print(section, key, values, flush=True)
      save_metrics(section, key, values)
    return

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
      for (section, key), values in new_user_evaluation(model, train, test, eval_users, catalogue_users).items():
        print(section, key, values, flush=True)
        save_metrics(section, key, values)

    del model
    gc.collect()

if __name__ == '__main__':
  main()
