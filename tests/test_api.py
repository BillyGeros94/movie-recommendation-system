from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import api
from content_model import ContentBasedRecommender
from item_model import ItemBasedCF
from uv_model import UVDecomposition

@pytest.fixture(scope='module')
def client(ratings, movies):
  uv = UVDecomposition(n_factors=4, std=0.1)
  uv.fit(ratings, movies, n_epochs=5, lr=0.02, batch_size=500)
  item = ItemBasedCF(n_neighbors=10, sim_shrinkage=5.0, block_size=16)
  item.fit(ratings, movies)
  content = ContentBasedRecommender()
  content.fit(ratings, movies)

  # The API serves the slim copies, as it does in production. Skipping the context manager skips file loading
  api.app.state.uv_model = uv.for_serving()
  api.app.state.item_model = item.for_serving()
  api.app.state.content_model = content.for_serving()
  return TestClient(api.app)

def test_item_endpoint_accepts_the_documented_body(client):
  response = client.post('/recommend/item', json={'movieIds': [2, 4, 6], 'n': 3})
  assert response.status_code == 200
  assert len(response.json()) == 3

def test_item_endpoint_rejects_only_unknown_seeds(client):
  assert client.post('/recommend/item', json={'movieIds': [999999]}).status_code == 422

def test_uv_endpoint_rejects_unknown_movies_and_bad_ratings(client):
  assert client.post('/recommend/uv', json={'rated_movies': [{'movieId': 999999, 'rating': 4}]}).status_code == 422
  assert client.post('/recommend/uv', json={'rated_movies': [{'movieId': 1, 'rating': 9}]}).status_code == 422
  assert client.post('/recommend/uv', json={'rated_movies': []}).status_code == 422

def test_content_endpoint_rejects_unknown_terms(client):
  assert client.post('/recommend/content', json={'genres': ['zzzzqqq']}).status_code == 422

def test_content_endpoint_returns_titles(client):
  response = client.post('/recommend/content', json={'genres': ['Comedy'], 'n': 2})
  assert response.status_code == 200
  assert set(response.json()[0]) == {'movieId', 'title'}

def test_genres_are_clean(client):
  assert 'Comedy' in client.get('/genres').json()

@pytest.mark.parametrize('query', ['(', '[', '*', '(2000', 'Movie 1 ('])
def test_search_treats_symbols_as_text(client, query):
  assert client.get('/movies/search', params={'query': query}).status_code == 200

def test_search_finds_titles_case_insensitively(client):
  assert client.get('/movies/search', params={'query': 'movie 12 (2000)'}).json()[0]['movieId'] == 12

def test_concurrent_requests_all_succeed_and_leave_models_unchanged(client):
  uv, item = api.app.state.uv_model, api.app.state.item_model
  q_shape, sim_shape = uv.Q.shape, item.sim.shape
  body = {'rated_movies': [{'movieId': 2, 'rating': 5}, {'movieId': 3, 'rating': 1}], 'n': 3}

  def call(i):
    if i % 2:
      return client.post('/recommend/uv', json=body)
    return client.post('/recommend/item', json={'movieIds': [2, 4], 'n': 3})

  with ThreadPoolExecutor(8) as pool:
    responses = list(pool.map(call, range(200)))

  assert {r.status_code for r in responses} == {200}
  assert len({r.text for r in responses[1::2]}) == 1
  assert uv.Q.shape == q_shape and item.sim.shape == sim_shape

def test_startup_explains_a_missing_model_file(tmp_path, monkeypatch):
  monkeypatch.setattr(api, 'MODEL_DIR', tmp_path)
  with pytest.raises(RuntimeError, match='train.py'):
    with TestClient(api.app):
      pass
