# Load Dependencies
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.content_model = joblib.load(MODEL_DIR / 'content_model.joblib')
    app.state.item_model = joblib.load(MODEL_DIR / 'item_model.joblib')
    app.state.uv_model = joblib.load(MODEL_DIR / 'uv_model.joblib')
    yield

app = FastAPI(lifespan=lifespan)

class ContentRequest(BaseModel):
    genres: List[str]
    tags: List[str]
    n: int = 10

class ItemRequest(BaseModel):
  movieId: int
  n: int = 10

class RatedMovie(BaseModel):
  movieId: int
  rating: float

class UVRequest(BaseModel):
  rated_movies: List[RatedMovie]
  n: int = 10

@app.get("/genres")
def get_genres():
    model = app.state.content_model
    genres = set()
    for g in model.movies_df['genres_str']:
        for genre in g.split():
            genres.add(genre)
    return sorted(list(genres))

@app.get("/tags")
def get_tags():
    model = app.state.content_model
    return sorted(list(model.vectorizer.vocabulary_.keys()))

@app.get("/movies/search")
def search_movies(query: str):
  model = app.state.item_model
  known_movie_ids = set(model.movie_idx.keys())
  results = model.movies_df[
    model.movies_df['title'].str.contains(query, case=False, na=False) &
    model.movies_df['movieId'].isin(known_movie_ids)
  ]
  return results[['movieId', 'title']].head(10).to_dict(orient='records')

@app.post("/recommend/content")
def content_recommend(request: ContentRequest):
  model = app.state.content_model

  metadata = ' '.join(request.genres) + ' ' + ' '.join(request.tags)
  u_vector = model.vectorizer.transform([metadata])

  fake_user_id = -1
  fake_user_idx = model.user_profiles_matrix.shape[0]

  model.user_idx_map[fake_user_id] = fake_user_idx
  model.user_profiles_matrix = vstack([model.user_profiles_matrix, u_vector])
  model.user_profiles_norms = np.append(model.user_profiles_norms, np.sqrt(u_vector.power(2).sum()))

  result = model.recommend(fake_user_id, request.n)

  del model.user_idx_map[fake_user_id]
  model.user_profiles_matrix = model.user_profiles_matrix[:-1]
  model.user_profiles_norms = model.user_profiles_norms[:-1]

  return result[['title']].to_dict(orient='records')

@app.post("/recommend/item")
def item_recommend(request: ItemRequest):
  model = app.state.item_model

  movie_col = model.movie_idx[request.movieId]
  movie_mean = model.movie_means[request.movieId]

  fake_user_id = -1
  fake_user_idx = model.r_matrix.shape[0]

  n_movies = model.r_matrix.shape[1]
  fake_r_row = csr_matrix(([5.0], ([0], [movie_col])), shape=(1, n_movies))
  fake_b_row = csr_matrix(([5.0 - movie_mean], ([0], [movie_col])), shape=(1, n_movies))

  model.user_idx[fake_user_id] = fake_user_idx
  model.user_inv_idx[fake_user_idx] = fake_user_id
  model.user_means[fake_user_id] = 5.0
  model.r_matrix = vstack([model.r_matrix, fake_r_row]).tocsr()
  model.r_csc = model.r_matrix.tocsc()
  model.b_matrix = vstack([model.b_matrix, fake_b_row]).tocsr()

  result = model.recommend(fake_user_id, request.n)

  del model.user_idx[fake_user_id]
  del model.user_inv_idx[fake_user_idx]
  del model.user_means[fake_user_id]
  model.r_matrix = model.r_matrix[:-1]
  model.b_matrix = model.b_matrix[:-1]
  model.r_csc = model.r_matrix.tocsc()

  return result[['title']].to_dict(orient='records')

@app.post("/recommend/uv")
def uv_recommend(request: UVRequest):
  model = app.state.uv_model

  rated_movie_indices = []
  ratings = []

  for rm in request.rated_movies:
    if rm.movieId in model.movie_idx:
      rated_movie_indices.append(model.movie_idx[rm.movieId])
      ratings.append(rm.rating)

  rated_movie_indices = np.array(rated_movie_indices)
  ratings = np.array(ratings)

  weights = ratings / ratings.sum()
  P_new = np.average(model.Q[rated_movie_indices], axis=0, weights=weights)

  fake_user_id = -1
  fake_user_idx = model.P.shape[0]

  model.user_idx[fake_user_id] = fake_user_idx
  model.P = np.vstack([model.P, P_new])

  fake_rows = pd.DataFrame({
    'userId': fake_user_id,
    'movieId': [rm.movieId for rm in request.rated_movies],
    'rating': [rm.rating for rm in request.rated_movies]
  })
  model.train_df = pd.concat([model.train_df, fake_rows], ignore_index=True)

  result = model.recommend(fake_user_id, request.n)

  del model.user_idx[fake_user_id]
  model.P = model.P[:-1]
  model.train_df = model.train_df[model.train_df['userId'] != fake_user_id]

  return result[['title']].to_dict(orient='records')