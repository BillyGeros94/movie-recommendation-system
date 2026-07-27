# Movie Recommendation System

Recommendation systems are among the most impactful applications of machine learning, powering content discovery across streaming platforms, e-commerce, and social media. At their core they solve a single problem: given a large catalogue and a user with unknown preferences over most of it, surface the items most likely to be relevant.

This project builds a movie recommendation engine from scratch on the [MovieLens 33M dataset](https://grouplens.org/datasets/movielens/), implementing four algorithms that each approach that problem differently. All models follow a consistent sklearn-style class interface, are evaluated on the same held-out test set, and are served through a FastAPI layer. A live demo of the best-performing model runs on Streamlit Cloud.

---

## The Four Approaches

**Content-Based Filtering** builds a profile for each user from the metadata of movies they have rated. Genres and genome tag scores (relevance ≥ 0.5) are combined into a per-movie corpus and vectorised with TF-IDF. A user profile is then the weighted sum of the TF-IDF vectors of rated movies, with mean-centred ratings as weights — so positively rated movies pull the profile toward their content and negatively rated ones push away. Recommendations are the unrated movies most similar to that profile by cosine similarity.

**User-Based Collaborative Filtering** identifies users with similar rating patterns and aggregates their preferences. Cosine similarity is computed between the target user and all others on the raw rating matrix. Only neighbours above a similarity threshold of 0.3 are used, capped at 100. Predictions are the target user's mean rating adjusted by a similarity-weighted average of neighbours' mean-centred ratings, correcting for individual rating scale differences.

**Item-Based Collaborative Filtering** computes similarity between items based on how users have co-rated them. For a target user-item pair the model retrieves items the user has already rated, computes cosine similarity between those and the target item, and produces a prediction as the target item's mean rating adjusted by the similarity-weighted average of the user's mean-centred ratings on similar items.

**UV Decomposition (Matrix Factorisation)** factorises the rating matrix into a user factor matrix P and an item factor matrix Q, each with 15 latent dimensions. Predicted ratings are dot products of the corresponding factor vectors. Both matrices are learned via mini-batch SGD with L2 regularisation, shuffling the data each epoch, with gradient clipping for stability. Training runs up to 50 epochs with early stopping when observed MSE falls below 0.60.

---

## Dataset

- **Source**: [MovieLens 33M](https://grouplens.org/datasets/movielens/) — GroupLens Research
- **Raw scale**: 33.8M ratings, 86K movies, 331K users
- **After co-filtering**: 26M ratings, 11,943 movies, 81,612 users

An iterative co-filtering procedure retains only users and movies with at least 100 ratings, applied alternately until the dataset stabilises. A single pass is insufficient because removing low-activity users can drop a movie below the threshold and vice versa. The filter removes cold-start entities with unreliable signal while preserving a dense, statistically meaningful core — retaining 77% of all ratings.

The utility matrix across the filtered population is 97.34% sparse, which is the fundamental challenge all four models must overcome.

**Split**: Per-user chronological 80/20 — the model sees each user's earlier ratings and is evaluated on their later ones, simulating realistic deployment conditions and preventing future leakage into training.

| Set | Ratings |
|---|---|
| Train | ~20.7M |
| Test | ~5.2M |

---

## Repository Structure

```
movie-recommendation-system/
├── data/
│   └── ReadMe.md                # Dataset download instructions
├── notebooks/
│   └── Recommender Report.ipynb # Full pipeline: EDA, training, evaluation
├── src/
│   ├── base_model.py            # Abstract base class (fit / predict / recommend)
│   ├── content_model.py         # Content-based filtering
│   ├── user_model.py            # User-based collaborative filtering
│   ├── item_model.py            # Item-based collaborative filtering
│   ├── uv_model.py              # UV decomposition
│   └── evaluation.py            # Batched evaluation framework
├── api/
│   ├── api.py                   # FastAPI application
│   ├── requirements.txt
│   └── *.joblib                 # Serialized models (git-ignored)
├── streamlit/
│   ├── app.py                   # Streamlit demo
│   ├── requirements.txt
│   └── uv_model_deploy.joblib   # Lightweight deploy artifact
└── presentation/
```

---

## Evaluation

The `RecommenderEvaluator` class handles prediction in batches of 100,000 pairs to keep memory tractable. Five metrics across two categories are reported on the held-out test set.

**Rating accuracy** — how closely predicted ratings match actual ratings at the individual pair level. RMSE penalises large errors more heavily; MAE treats all errors equally and is more interpretable as an average absolute deviation in rating units.

**Ranking quality** — how well each model orders items for a given user. Precision@10 is the fraction of the top-10 recommended items that are relevant (rating ≥ 3.5). Recall@10 is the fraction of all relevant items that appear in the top-10. nDCG@10 extends precision by accounting for position — a relevant item ranked first contributes more than one ranked tenth — and normalises against the ideal ranking.

Full numerical results are reported in `notebooks/Recommender Report.ipynb`. Across all five metrics the consistent ranking is:

**UV Decomposition > User-Based > Item-Based > Content-Based**

UV leads on both accuracy and ranking simultaneously, which is not guaranteed — a model can predict ratings well without ranking well. That it dominates both dimensions indicates the latent factors capture preference structure relevant to both tasks. The high absolute values across all models (Precision@10 above 0.76 everywhere) reflect the density of the filtered population where every model has abundant signal. Content-based being last is structural rather than a failure: it predicts from metadata alone with no knowledge of how users actually rated.

---

## API

The FastAPI layer serves three recommendation endpoints and two utility endpoints. Models are loaded once at startup via joblib and held in application state.

**Run locally:**
```bash
cd api
pip install -r requirements.txt
uvicorn api:app --reload
```

Interactive docs available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/genres` | All available genres |
| `GET` | `/tags` | All genome tags in the vocabulary |
| `GET` | `/movies/search?query=` | Title search |
| `POST` | `/recommend/content` | Content-based recommendations |
| `POST` | `/recommend/item` | Item-based CF recommendations |
| `POST` | `/recommend/uv` | Matrix factorisation recommendations |

**Content-based example:**
```json
POST /recommend/content
{
  "genres": ["Action", "Thriller"],
  "tags": ["suspense", "twist ending"],
  "n": 10
}
```

**UV example:**
```json
POST /recommend/uv
{
  "rated_movies": [
    {"movieId": 318, "rating": 5.0},
    {"movieId": 296, "rating": 4.5}
  ],
  "n": 10
}
```

---

## Streamlit Demo

An interactive demo of the UV model runs on Streamlit Cloud. The deployed artifact is a lightweight dict containing only the Q matrix and movie mappings (4.6MB), stored directly in the repository. A temporary user vector is constructed on the fly from the input ratings via weighted averaging of Q rows — no retraining or user history required.

Users search for movies by title, rate at least 5, and receive 10 personalised recommendations.

**Live demo:** *(link here once deployed)*

---

## Key Design Decisions

- **Iterative co-filtering** rather than a single-pass threshold, because removing users can cascade into movie removals and vice versa.
- **Temporal split** rather than random, to reflect realistic deployment where future ratings are unseen at training time.
- **Sparse matrix storage** (CSR/CSC) throughout the CF models to keep 26M ratings in memory without materialising a dense 81K × 12K matrix.
- **Batched prediction** in the evaluator to keep memory tractable on 5.2M test pairs.
- **Gradient clipping** in UV training to prevent exploding updates at 26M observations per epoch.
- **Fake user injection** in the API endpoints to serve cold recommendations without modifying trained model state, with full cleanup after each request.