# Load Dependencies
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
import sys

import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

# Set model directory path
MODEL_DIR = Path(__file__).parent / 'models'
MODEL_FILES = {'content_model': 'content.joblib', 'item_model': 'item.joblib', 'uv_model': 'uv.joblib'}

# Load all models once at startup and store in application state. Requests only read them
@asynccontextmanager
async def lifespan(app: FastAPI):
  for name, filename in MODEL_FILES.items():
    path = MODEL_DIR / filename
    if not path.exists():
      raise RuntimeError(f'Missing model file {path}. Run "python train.py" from the repository root to create it.')
    setattr(app.state, name, joblib.load(path))
  yield

app = FastAPI(title='Movie Recommendation API', lifespan=lifespan)

# Request Schemas
class ContentRequest(BaseModel):
  genres: List[str] = []
  tags: List[str] = []
  n: int = Field(10, ge=1, le=100)

class ItemRequest(BaseModel):
  movieIds: List[int] = Field(min_length=1)
  n: int = Field(10, ge=1, le=100)

class RatedMovie(BaseModel):
  movieId: int
  rating: float = Field(ge=0.5, le=5.0)

class UVRequest(BaseModel):
  rated_movies: List[RatedMovie] = Field(min_length=1)
  n: int = Field(10, ge=1, le=100)

def to_response(recommendations):
  return recommendations[['movieId', 'title']].to_dict(orient='records')

# Return a sorted list of all genres the content model knows
@app.get('/genres')
def get_genres():
  return app.state.content_model.genres

# Return a sorted list of all terms in the TF-IDF vocabulary
@app.get('/tags')
def get_tags():
  return sorted(app.state.content_model.vectorizer.vocabulary_.keys())

# Search for movies by title substring, treating the query as plain text
@app.get('/movies/search')
def search_movies(query: str = Query(min_length=1)):
  needle = query.lower()
  titles = app.state.item_model.titles
  matches = [{'movieId': m, 'title': t} for m, t in titles.items() if needle in t.lower()]
  return matches[:10]

# Content-based recommendations
@app.post('/recommend/content')
def content_recommend(request: ContentRequest):
  result = app.state.content_model.recommend_from_text(' '.join(request.genres + request.tags), request.n)
  if result.empty:
    raise HTTPException(422, 'None of the genres or tags are in the model vocabulary. See /genres and /tags.')
  return to_response(result)

# Item-based recommendations
@app.post('/recommend/item')
def item_recommend(request: ItemRequest):
  result = app.state.item_model.recommend_from_seeds(request.movieIds, request.n)
  if result.empty:
    raise HTTPException(422, 'None of the movieIds are in the model. Use /movies/search to find valid IDs.')
  return to_response(result)

# Matrix factorisation recommendations
@app.post('/recommend/uv')
def uv_recommend(request: UVRequest):
  ratings = {movie.movieId: movie.rating for movie in request.rated_movies}
  result = app.state.uv_model.recommend_new_user(ratings, request.n)
  if result.empty:
    raise HTTPException(422, 'None of the rated movies are in the model. Use /movies/search to find valid IDs.')
  return to_response(result)
